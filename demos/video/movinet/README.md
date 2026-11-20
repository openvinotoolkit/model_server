# Standalone streaming MoViNet POC

This POC tests the local `movinet_a0_stream.xml` export independently first,
then connects it to native OVMS WebRTC video ingestion. Video processing is
opt-in; ordinary audio calls keep their existing ASR path.

## Run

Run from the repository root with a Python environment containing the packages
in `requirements.txt`. The existing repository `.venv` was used for this test.
The model was found at `$HOME/models/OpenVINO/movinet_openvino`, outside the
checkout's `models` directory.

Download the sample and class labels used by the
[official tutorial](https://github.com/tensorflow/models/blob/master/official/projects/movinet/movinet_tutorial.ipynb):

```bash
mkdir -p local/movinet
curl -fL https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/jumpingjack.gif \
    -o local/movinet/jumpingjack.gif
curl -fL https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt \
    -o local/movinet/kinetics_600_labels.txt

.venv/bin/python -m demos.video.movinet.standalone \
    --model "$HOME/models/OpenVINO/movinet_openvino/movinet_a0_stream.xml" \
    --video local/movinet/jumpingjack.gif \
    --labels local/movinet/kinetics_600_labels.txt \
    --verify --expect-label "jumping jacks"
```

The sample footage is by
[Coach Bobby Bluford](https://www.youtube.com/watch?v=-AxHpj-EuPg), shared under
CC-BY according to the upstream tutorial.

The command prints periodic top-five predictions and a final JSON summary.
`--verify` replays the clip after resetting, interleaves a second independent
session, and checks that preserving history changes the output versus resetting
on every frame. A wrong expected top label fails the command.

## Verified result

On an Intel Xeon CPU Max 9480, using two inference threads and FP32 with
OpenVINO `2026.5.0-23160-6db412d0a27`:

- All 13 GIF frames decoded and ran successfully.
- Frame 10: **jumping jacks, 98.96%**.
- Final frame: **jumping jacks, 99.04%**.
- Reset replay and independent-session checks passed.
- Resetting before every frame instead predicted "passing american football
  (not in game)" on the last frame. Maximum stateful/stateless probability
  difference over the clip: `0.992565`.
- A separate comparison against the local original TFLite artifact, using
  `tf.lite.Interpreter` with two threads and identical input tensors, passed on
  every frame. Maximum absolute probability difference: `2.03e-6`.
- Repeating the GIF 20 times without resetting (260 frames) produced finite,
  normalized predictions throughout; final "jumping jacks" probability: **97.21%**.
- One standalone run measured approximately **5.48 ms/frame** mean and **7.21 ms**
  p95, including preprocessing, inference, state updates, and softmax.
  This tiny sample is a smoke-test timing, not a representative benchmark;
  decoding, networking, and codec costs are excluded.

## Streaming contract of this particular export

This is a causal, frame-by-frame action classifier, not a person detector.
It produces one whole-frame distribution over 600 Kinetics classes.

The local TFLite and IR artifacts have a custom flattened signature, unlike the
named-state signature in the TensorFlow tutorial. Do not substitute an upstream
TFLite model and assume the same port mapping:

| Inputs | Meaning | Update after each inference |
|---|---|---|
| `args_0` | One RGB image, FP32 `[1, 3, 172, 172]`, scaled to `[0, 1]` | OpenCV BGR to RGB, bilinear resize, transpose to NCHW |
| `args_1` through `args_28` | Eleven temporal histories, with depths `(2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4)` | Shift oldest out; append the corresponding `output_1` through `output_11` |
| `args_29` through `args_44` | Sixteen cumulative pooling sums | Add `output_12` through `output_27`, respectively |
| `args_45` | Pooling reciprocal | Set to `1 / (frames_already_processed + 1)` |
| `args_46` | Temporal-history output scale | Set to `1` |

`output_0` is `[1, 600]` logits; apply softmax to obtain probabilities.
Initialize histories and pooling sums to zero.

`MovinetStream` owns an infer request and state per session. A compiled model
can be shared across sessions; the mutable stream object cannot be used
concurrently. State memory stays bounded as the stream grows. Predictions use
cumulative pooling, **not a rolling window**: reset on a new clip/session, or at
explicit application-selected boundaries if older activities should be forgotten.
This describes the standalone Python reference; the native WebRTC path below
uses rolling pooling.

## Native WebRTC POC

The native path is:

```text
browser / aiortc -> VP8 RTP -> libdatachannel -> RFC 7741 depacketization
  -> libvpx -> RGB ImageFrame -> LiveGraphSession (video.classification)
  -> MovinetCalculator (resize, inference, state) -> oai-events data channel
```

Build inside a builder container with the checkout mounted at `/ovms`, models
mounted at `/ovms/models`, and the chosen REST port plus UDP media range
reachable from your client. These commands use the existing nested Bazel cache:

```bash
cd /ovms
REST_PORT=9112 ./demos/video/movinet/run_server.sh SERVER_IP
```

The runner builds `//src:ovms` with `--define BUILD_WITH_WEBRTC=1`. VP8 decoding
builds a pinned static libvpx dependency through Bazel; no system codec
installation or Dockerfile changes are needed.

The runner serves the `movinet` graph and sets `OVMS_WEBRTC_VIDEO_GRAPH=movinet`.
It generates a runtime graph/config under `local/movinet/runtime` from
[graph.pbtxt](graph.pbtxt) and [config.json](config.json). Override its asset
paths with `MOVINET_MODEL_PATH` and `MOVINET_LABELS_PATH`. When launching OVMS
directly, use `OVMS_WEBRTC_VIDEO_GRAPH=movinet` with this demo config and adjust
the graph's model/label paths as needed.

The compiled model is loaded once when the graph is loaded. Each call opens a
managed `LiveGraphSession` and owns an independent calculator, infer request,
temporal history, pooling sums, and VP8 decoder. Both MoViNet and YOLO now use
this live-graph path; the decoder contains no model inference logic.
The old `OVMS_WEBRTC_VIDEO_MODEL` / `OVMS_WEBRTC_VIDEO_LABELS` direct-inference
path has been removed. Without `OVMS_WEBRTC_VIDEO_GRAPH`, audio behavior is
unchanged.

### Browser camera client

From the checkout on the client machine:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory demos/video/movinet
```

Open `http://localhost:8000`, set the reachable OVMS URL, and click **Start**.
The page requests a 640x480 camera stream at approximately 8 FPS and negotiates
VP8 explicitly. **Stop / reset history** closes the call; starting again creates
new model state. This simple page targets a local POC server without API-key
authentication. Camera access requires localhost or HTTPS. HTTPS pages also
require an HTTPS signaling endpoint.

For video only, use the served graph name `movinet` as the session model. For simultaneous ASR,
include both the MoViNet graph and your existing speech-to-text graph in the
server configuration, enter the served ASR model name, and enable the microphone
checkbox. Video uses `OVMS_WEBRTC_VIDEO_GRAPH`; audio uses the existing managed
speech-to-text graph.

The server uses the existing multipart `POST /v1/realtime/calls` signaling,
`oai-events` channel, call `Location`, and hangup endpoint. It accepts one VP8
track per call and filters the negotiated video codecs to VP8. Video events are
a POC extension, not an OpenAI-standard video API:

```json
{
  "type": "conversation.item.input_video_classification.completed",
  "event_id": "event_20",
  "item_id": "video_3_13",
  "frame": 13,
  "rtp_timestamp": 3715954780,
  "label": "jumping jacks",
  "score": 0.99219,
  "transcript": "jumping jacks"
}
```

There is one event per graph-processed frame. The bounded graph queue drops
decoded frames under backpressure; frame counters need not be consecutive.
`score` is the top class's
softmax probability, not a calibrated confidence or detection threshold.
`transcript` mirrors the label for simple text consumers; existing audio event
types are unchanged. Standard `error` events report decoder/inference failures.

Native sessions use a **32-frame rolling pooling window** (about four seconds
at 8 FPS). Before each inference, the oldest pooling increments are removed;
the graph averages the remaining increments plus the current frame using
`1 / min(processed_frames + 1, 32)`. Temporal convolution histories are preserved.
This is approximate recent-context classification, not an isolated-clip replay:
retained features and temporal states can still reflect earlier frames. The
window is frame-based, so its duration varies with the actual processed FPS.
Each session preallocates a ring buffer for the 16 pooling outputs; there is no
extra inference or per-frame history allocation. Stop/start resets all history.

At INFO log level, each active stream logs statistics approximately once per
second (on the next prediction):

```text
MoViNet stream=0x... prediction_fps=8.00 avg_inference_ms=12.34 window_frames=8 window_seconds=1.00 total_frames=16 pooling_window_frames=32
```

`prediction_fps` is completed predictions per wall-clock second in that window,
including time waiting for incoming frames, not the model's maximum throughput.
An unchanged label still counts as a new prediction. `avg_inference_ms` measures
only OpenVINO inference, excluding decoding, preprocessing and state updates.
`stream` identifies the stream instance; counters reset for each new call.
No periodic log is emitted while no predictions arrive.

For remote clients, publish the chosen REST TCP port and media UDP range, and
configure `OVMS_WEBRTC_ANNOUNCED_IP` as described in the
[audio WebRTC documentation](../../audio/voxtral_webrtc/README.md).

### Native validation

Before the migration, the following build and focused tests passed in the existing builder:

```bash
bazel --output_user_root=/root/.cache/bazel/bazel build --jobs=8 \
    --define BUILD_WITH_WEBRTC=1 --define BUILD_WITH_EXPERIMENTAL_VOXTRAL=1 //src:ovms

bazel --output_user_root=/root/.cache/bazel/bazel test --jobs=8 \
    --define BUILD_WITH_WEBRTC=1 --define BUILD_WITH_EXPERIMENTAL_VOXTRAL=1 \
    --test_output=errors //src/webrtc:webrtc_peer_connection_test \
    //src/webrtc:webrtc_session_controller_test
```

The focused targets included VP8 extended descriptors, sequence-number wrap,
missing fragments, malformed descriptors, existing audio transport, and
controller tests. Model-backed ASR tests require their documented environment
variables and are skipped when these are absent.

After migration to `LiveGraphSession`, the server build and these focused
targets passed:

```bash
bazel --output_user_root=/root/.cache/bazel/bazel test --jobs=8 \
    --define BUILD_WITH_WEBRTC=1 --test_output=errors \
    //src:movinet_calculator_test //src/webrtc:webrtc_session_controller_test \
    //src/webrtc:webrtc_peer_connection_test
```

The calculator tests use a small generated IR to verify rolling-pooling
boundaries, retained temporal history, independent simultaneous sessions,
fresh state after restart, and resource validation. The model-backed aiortc
tests below also passed through the live graph, including the 72-frame rolling
reference comparison; classification event fields and scores were unchanged.
All seven shared live-graph regression tests also passed:

```bash
bazel --output_user_root=/root/.cache/bazel/bazel test --jobs=8 \
    --define BUILD_WITH_WEBRTC=1 --test_output=errors \
    --test_filter='LiveGraphSessionTest.*:LiveGraphStreamCodecTest.*:LiveGraphProfileTest.*' \
    //src:ovms_test
```

The controller target was also rerun successfully after migration with
`--test_env=OVMS_WHISPER_MODEL_PATH=/ovms/models/OpenVINO/whisper-tiny-int4-ov`
and `--test_env=OVMS_VOXTRAL_TEST_WAV=/ovms/local/movinet/asr-regression.wav`,
running both real-model Whisper WebRTC cases (silence-triggered and explicit
commit). That WAV was decoded from the existing
`references/SayLargestAnimal.webm` sample to mono signed 16-bit PCM at 16 kHz.
The three model-backed Voxtral cases were not run.

With the test server running, this command passed from the checkout:

```bash
OVMS_VIDEO_URL=http://127.0.0.1:6666 .venv/bin/python -m pytest \
    --rootdir=tests/webrtc_e2e -v -s tests/webrtc_e2e/test_webrtc_video.py
```

Port 6666 was the existing builder's published test port; change the URL for
your deployment. Prefer another port for browser access because browsers may
block port 6666.

Both `sendonly` and browser-like `sendrecv` offers passed: **all 13 frames
produced data-channel predictions**, with final **jumping jacks, 99.22%** after
actual VP8 encoding, RTP transport, depacketization, and decoding.
The browser UI requires a manual camera check; automated validation uses aiortc.
H264-only and multi-video-track offers are also tested and rejected with HTTP 400.
The rolling-pooling test additionally needs the standalone Python dependencies
and access to the server's model and labels (`WEBRTC_MOVINET_MODEL` and
`WEBRTC_MOVINET_LABELS` override the checkout-relative defaults). It sends 72
pre-encoded VP8 frames with a scene change, crosses two ring-buffer boundaries,
and compares every prediction against an independent reference that recomputes
recent-increment sums instead of subtracting from a ring buffer.

### Deliberate POC limitations

- CPU-only inference, fixed custom A0 signature; one even-sized I420 video track
  per call, with each dimension limited to 1920 pixels.
- No production jitter buffer, retransmission, or keyframe-recovery policy.
  Incomplete frames are dropped with a warning; decoder failures stop video
  and send an error. Reconnect after an error.
- The encoded input queue holds at most 16 frames (2 MiB maximum per frame).
  Overflow stops video with an error instead of silently corrupting temporal
  history or blocking the audio/transport thread.
- Prediction events can be discarded by the existing bounded outbound queue
  under backpressure. Inference and VP8 decoding run off the transport thread.
- Rolling pooling retains 32 frames, but temporal states and retained features
  can carry older context. There is no object localization, multi-person
  separation, or automatic action boundary detection.
- The served graph selects the model and labels via calculator options;
  `OVMS_WEBRTC_VIDEO_GRAPH` selects one video graph for all calls on a server.
