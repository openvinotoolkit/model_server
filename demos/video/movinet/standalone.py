# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frame-by-frame inference for the local MoViNet-A0 custom streaming export."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# This export unrolls temporal kernels into separate oldest-to-newest frames.
TEMPORAL_DEPTHS = (2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4)
IMAGE_INPUT = 0
POOL_INPUT_START = 1 + sum(TEMPORAL_DEPTHS)
POOL_OUTPUT_START = 1 + len(TEMPORAL_DEPTHS)
POOL_COUNT = 16
POOL_SCALE_INPUT = POOL_INPUT_START + POOL_COUNT
HISTORY_SCALE_INPUT = POOL_SCALE_INPUT + 1
PIXEL_MAX = 255.0
MILLISECONDS_PER_SECOND = 1000


class MovinetStream:
    """Own one session's temporal buffers and cumulative pooling state.

    The compiled model can be shared, but each video stream needs its own instance.
    Input frames are OpenCV BGR uint8 images. Results are Kinetics-600 probabilities.
    """

    def __init__(self, compiled_model):
        self.model = compiled_model
        if len(self.model.inputs) != HISTORY_SCALE_INPUT + 1:
            raise ValueError("Expected the custom MoViNet export with 47 inputs")
        if len(self.model.outputs) != POOL_OUTPUT_START + POOL_COUNT:
            raise ValueError("Expected the custom MoViNet export with 28 outputs")
        self.inputs = [
            np.zeros(port.shape, dtype=np.float32) for port in self.model.inputs
        ]
        self.request = self.model.create_infer_request()
        self.height, self.width = self.model.input(IMAGE_INPUT).shape[-2:]
        self.reset()

    def reset(self):
        """Forget all history, including cumulative pooling, at a stream boundary."""
        for value in self.inputs:
            value.fill(0)
        self.frames = 0

    def infer(self, bgr_frame):
        """Consume exactly one frame and return this stream's current prediction."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self.inputs[IMAGE_INPUT][0] = resized.transpose(2, 0, 1) / PIXEL_MAX
        self.inputs[POOL_SCALE_INPUT].fill(1.0 / (self.frames + 1))
        self.inputs[HISTORY_SCALE_INPUT].fill(1)
        self.request.infer(dict(enumerate(self.inputs)))
        logits = self.request.get_output_tensor(0).data[0].copy()
        if not np.isfinite(logits).all():
            raise RuntimeError(f"Non-finite model output at frame {self.frames}")

        start = 1
        for output, depth in enumerate(TEMPORAL_DEPTHS, start=1):
            for index in range(start, start + depth - 1):
                np.copyto(self.inputs[index], self.inputs[index + 1])
            np.copyto(
                self.inputs[start + depth - 1],
                self.request.get_output_tensor(output).data,
            )
            start += depth
        for offset in range(POOL_COUNT):
            self.inputs[POOL_INPUT_START + offset] += self.request.get_output_tensor(
                POOL_OUTPUT_START + offset
            ).data
        self.frames += 1
        scores = np.exp(logits - logits.max())
        return scores / scores.sum()


def top_predictions(scores, labels, count=5):
    """Format the highest scoring classes for JSON/data-channel output."""
    return [
        {"label": labels[index], "score": float(scores[index])}
        for index in np.argsort(scores)[-count:][::-1]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--verify", action="store_true",
                        help="Check reset replay, independent sessions, and stateful vs reset inference")
    parser.add_argument("--expect-label")
    args = parser.parse_args()
    if args.threads < 1 or args.report_every < 1:
        parser.error("--threads and --report-every must be positive")

    labels = args.labels.read_text().splitlines()
    core = ov.Core()
    model = core.compile_model(str(args.model), "CPU", {
        "INFERENCE_NUM_THREADS": args.threads,
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f32",
    })
    if len(labels) != model.output(0).shape[-1]:
        raise ValueError("Label count does not match model output")
    stream = MovinetStream(model)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    durations = []
    frames = []
    predictions = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            scores = stream.infer(frame)
            durations.append(time.perf_counter() - started)
            if args.verify:
                frames.append(frame)
                predictions.append(scores)
            if stream.frames == 1 or stream.frames % args.report_every == 0:
                print(json.dumps({"frame": stream.frames,
                                  "predictions": top_predictions(scores, labels)}), flush=True)
    finally:
        capture.release()
    if not durations:
        raise RuntimeError("Video contains no decodable frames")

    summary = {
        "frames": stream.frames,
        "final": top_predictions(scores, labels),
        "mean_frame_ms": float(np.mean(durations) * MILLISECONDS_PER_SECOND),
        "p95_frame_ms": float(np.percentile(durations, 95) * MILLISECONDS_PER_SECOND),
        "processing_fps": len(durations) / sum(durations),
        "openvino": ov.__version__,
        "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
    }
    if args.verify:
        stream.reset()
        independent = MovinetStream(model)
        reset_each_frame = MovinetStream(model)
        stateless_scores = []
        for frame, expected in zip(frames, predictions):
            np.testing.assert_allclose(stream.infer(frame), expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(independent.infer(frame), expected, rtol=1e-5, atol=1e-6)
            reset_each_frame.reset()
            stateless_scores.append(reset_each_frame.infer(frame))
        difference = float(np.max(np.abs(np.array(predictions) - stateless_scores)))
        if difference <= 1e-6:
            raise AssertionError("State feedback did not affect predictions")
        summary["verification"] = {
            "reset_replay": "passed",
            "independent_session": "passed",
            "stateful_vs_stateless_max_probability_difference": difference,
            "reset_each_frame_final": top_predictions(stateless_scores[-1], labels),
        }
    print(json.dumps({"summary": summary}), flush=True)
    if args.expect_label and summary["final"][0]["label"] != args.expect_label:
        raise AssertionError(f"Expected top prediction {args.expect_label!r}")


if __name__ == "__main__":
    main()
