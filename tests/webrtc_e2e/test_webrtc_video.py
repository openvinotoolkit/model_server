# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.

"""Native VP8 -> MoViNet -> oai-events smoke test; requires an opt-in OVMS server."""

import asyncio
import json
import os
from collections import deque
from pathlib import Path

import pytest
import requests
import av
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer


VIDEO_URL = os.environ.get("OVMS_VIDEO_URL", "").rstrip("/")
VIDEO_FILE = Path(os.environ.get("WEBRTC_VIDEO_FILE", "local/movinet/jumpingjack.gif"))
CLASSIFICATION_EVENT = "conversation.item.input_video_classification.completed"
MIN_EXPECTED_SCORE = 0.9
SIGNALING_TIMEOUT_SECONDS = 30
PREDICTION_TIMEOUT_SECONDS = 20


@pytest.mark.skipif(not VIDEO_URL, reason="Set OVMS_VIDEO_URL to a server with the MoViNet POC enabled")
@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["sendonly", "sendrecv"])
async def test_native_vp8_movinet_classification(direction):
    assert VIDEO_FILE.is_file(), f"Download the upstream test sample first: {VIDEO_FILE}"
    with av.open(str(VIDEO_FILE)) as video:
        expected_frames = sum(1 for _ in video.decode(video=0))
    assert expected_frames > 0
    peer = RTCPeerConnection()
    player = MediaPlayer(str(VIDEO_FILE))
    channel = peer.createDataChannel("oai-events")
    predictions = []
    errors = []
    received_prediction = asyncio.Event()
    http = requests.Session()
    http.trust_env = False
    location = None

    @channel.on("message")
    def on_message(raw):
        event = json.loads(raw)
        if event["type"] == "error":
            errors.append(event)
            received_prediction.set()
        if event["type"] == CLASSIFICATION_EVENT:
            predictions.append(event)
            if event["frame"] >= expected_frames:
                received_prediction.set()

    try:
        transceiver = peer.addTransceiver(player.video, direction=direction)
        transceiver.setCodecPreferences([
            codec for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.lower() == "video/vp8"
        ])
        await peer.setLocalDescription(await peer.createOffer())
        response = await asyncio.to_thread(
            http.post, f"{VIDEO_URL}/v1/realtime/calls",
            files={
                "sdp": (None, peer.localDescription.sdp),
                "session": (None, json.dumps({"type": "transcription", "model": "movinet"})),
            },
            timeout=SIGNALING_TIMEOUT_SECONDS,
        )
        assert response.status_code == 200, response.text
        location = response.headers["Location"]
        assert "VP8/90000" in response.text
        assert "H264/90000" not in response.text
        await peer.setRemoteDescription(RTCSessionDescription(sdp=response.text, type="answer"))
        await asyncio.wait_for(received_prediction.wait(), timeout=PREDICTION_TIMEOUT_SECONDS)
        assert not errors, errors
        assert predictions
        assert predictions[-1]["frame"] == expected_frames
        assert predictions[-1]["label"] == "jumping jacks"
        assert predictions[-1]["score"] >= MIN_EXPECTED_SCORE
        assert all(0 <= p["score"] <= 1 and p["transcript"] == p["label"] for p in predictions)
        assert all(a["frame"] < b["frame"] for a, b in zip(predictions, predictions[1:]))
        print(json.dumps({"predictions_received": len(predictions), "last_prediction": predictions[-1]}))
    finally:
        if player.video:
            player.video.stop()
        await peer.close()
        if location:
            response = await asyncio.to_thread(
                http.post, f"{VIDEO_URL}{location}/hangup", timeout=SIGNALING_TIMEOUT_SECONDS,
            )
            assert response.status_code in (200, 204), response.text
        http.close()


@pytest.mark.skipif(not VIDEO_URL, reason="Set OVMS_VIDEO_URL to a server with the MoViNet POC enabled")
@pytest.mark.asyncio
async def test_native_rolling_pooling_matches_recent_increment_sums(tmp_path):
    ov = pytest.importorskip("openvino")
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    from demos.video.movinet.standalone import (
        MovinetStream, POOL_COUNT, POOL_INPUT_START, POOL_OUTPUT_START,
    )

    window_frames = 32
    frame_count = 2 * window_frames + 8
    model_path = Path(os.environ.get(
        "WEBRTC_MOVINET_MODEL", "models/OpenVINO/movinet_openvino/movinet_a0_stream.xml",
    ))
    labels_path = Path(os.environ.get(
        "WEBRTC_MOVINET_LABELS", "local/movinet/kinetics_600_labels.txt",
    ))
    assert model_path.is_file(), f"Set WEBRTC_MOVINET_MODEL to the server's model: {model_path}"
    labels = labels_path.read_text().splitlines()
    model = ov.Core().compile_model(str(model_path), "CPU", {
        "INFERENCE_NUM_THREADS": 2, "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
    })
    reference = MovinetStream(model)
    cumulative = MovinetStream(model)
    history = deque(maxlen=window_frames - 1)
    expected = []
    cumulative_scores = []
    black = np.zeros((96, 96, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    video_path = tmp_path / "scene-change.ivf"
    with av.open(str(video_path), mode="w", format="ivf") as output:
        stream = output.add_stream("libvpx", rate=30)
        stream.width = black.shape[1]
        stream.height = black.shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"deadline": "realtime", "lag-in-frames": "0"}
        for index in range(frame_count):
            image = black if index < window_frames else white
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(video_path)) as video:
        # Match native OpenCV conversion, using the exact VP8 frames sent without re-encoding.
        images = [
            cv2.cvtColor(frame.to_ndarray(format="yuv420p"), cv2.COLOR_YUV2BGR_I420)
            for frame in video.decode(video=0)
        ]
    assert len(images) == frame_count
    for index, image in enumerate(images):
        # Recompute sums independently instead of duplicating native ring-buffer subtraction.
        for state in range(POOL_COUNT):
            accumulator = reference.inputs[POOL_INPUT_START + state]
            accumulator.fill(0)
            for increments in history:
                accumulator += increments[state]
        reference.frames = min(index, window_frames - 1)
        expected.append(reference.infer(image))
        history.append([
            reference.request.get_output_tensor(POOL_OUTPUT_START + state).data.copy()
            for state in range(POOL_COUNT)
        ])
        cumulative_scores.append(cumulative.infer(image))
    expected = np.array(expected)
    best = expected.argmax(axis=1)
    expected_scores = expected[np.arange(frame_count), best]
    cumulative_scores = np.array(cumulative_scores)
    # Ensure this sequence distinguishes rolling pooling from the old cumulative behavior.
    score_rtol = 1e-3
    score_atol = 1e-4
    assert np.any(best != cumulative_scores.argmax(axis=1)) or not np.allclose(
        expected_scores, cumulative_scores.max(axis=1), rtol=score_rtol, atol=score_atol,
    )

    peer = RTCPeerConnection()
    player = MediaPlayer(str(video_path), decode=False)
    track = player.video
    assert track is not None
    channel = peer.createDataChannel("oai-events")
    predictions = []
    errors = []
    completed = asyncio.Event()
    http = requests.Session()
    http.trust_env = False
    location = None

    @channel.on("message")
    def on_message(raw):
        event = json.loads(raw)
        if event["type"] == "error":
            errors.append(event)
            completed.set()
        elif event["type"] == CLASSIFICATION_EVENT:
            predictions.append(event)
            if event["frame"] == frame_count:
                completed.set()

    try:
        transceiver = peer.addTransceiver(track, direction="sendonly")
        transceiver.setCodecPreferences([
            codec for codec in RTCRtpSender.getCapabilities("video").codecs
            if codec.mimeType.lower() == "video/vp8"
        ])
        await peer.setLocalDescription(await peer.createOffer())
        response = await asyncio.to_thread(
            http.post, f"{VIDEO_URL}/v1/realtime/calls",
            files={
                "sdp": (None, peer.localDescription.sdp),
                "session": (None, json.dumps({"type": "transcription", "model": "movinet"})),
            },
            timeout=SIGNALING_TIMEOUT_SECONDS,
        )
        assert response.status_code == 200, response.text
        location = response.headers["Location"]
        await peer.setRemoteDescription(RTCSessionDescription(sdp=response.text, type="answer"))
        await asyncio.wait_for(completed.wait(), timeout=PREDICTION_TIMEOUT_SECONDS)
        assert not errors, errors
        assert [p["frame"] for p in predictions] == list(range(1, frame_count + 1))
        assert [p["label"] for p in predictions] == [labels[index] for index in best]
        np.testing.assert_allclose(
            [p["score"] for p in predictions], expected_scores,
            rtol=score_rtol, atol=score_atol,
        )
    finally:
        track.stop()
        await peer.close()
        if location:
            response = await asyncio.to_thread(
                http.post, f"{VIDEO_URL}{location}/hangup", timeout=SIGNALING_TIMEOUT_SECONDS,
            )
            assert response.status_code in (200, 204), response.text
        http.close()


@pytest.mark.skipif(not VIDEO_URL, reason="Set OVMS_VIDEO_URL to a server with the MoViNet POC enabled")
@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", ["h264", "multiple-tracks"])
async def test_rejects_unsupported_video_offer(unsupported):
    peer = RTCPeerConnection()
    http = requests.Session()
    http.trust_env = False
    try:
        peer.createDataChannel("oai-events")
        for _ in range(2 if unsupported == "multiple-tracks" else 1):
            transceiver = peer.addTransceiver("video", direction="sendonly")
            mime_type = "video/h264" if unsupported == "h264" else "video/vp8"
            codecs = [
                codec for codec in RTCRtpSender.getCapabilities("video").codecs
                if codec.mimeType.lower() == mime_type
            ]
            assert codecs
            transceiver.setCodecPreferences(codecs)
        await peer.setLocalDescription(await peer.createOffer())
        response = await asyncio.to_thread(
            http.post, f"{VIDEO_URL}/v1/realtime/calls",
            files={
                "sdp": (None, peer.localDescription.sdp),
                "session": (None, json.dumps({"type": "transcription", "model": "movinet"})),
            },
            timeout=SIGNALING_TIMEOUT_SECONDS,
        )
        assert response.status_code == 400, response.text
    finally:
        await peer.close()
        http.close()
