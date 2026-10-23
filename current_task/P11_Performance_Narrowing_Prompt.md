# P11 - Narrow Video Inference Performance Investigation

## Goal

Explain the measured gap between pure OpenVINO D3D11 remote-tensor inference, direct OVMS C-API inference, and the unary OVMS MediaPipe file graph. Work from the established measurements below; do not rerun broad RTSP/camera experiments.

The question is not whether remote input works. It does. The question is where the remaining per-frame time is spent.

## Established Measurements

All measurements use `recording_3_raw.avi`, 672x384, Intel iGPU, detector `face_detection_adas`, `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, and four infer-request objects / OVMS `nireq=4`.

| Layer | Mean wall time | Equivalent FPS |
| --- | ---: | ---: |
| D3D11 decode-only file loop | 4.036 ms/frame | 247.76 |
| Pure OpenVINO D3D11 remote tensors | 7.226 ms/frame | 138.39 |
| Direct D3D11 plus OVMS C-API | 8.406 ms/frame | 118.96 |
| Unary OVMS MediaPipe graph request | 11.698 ms/frame | 85.48 |

Derived gaps:

- C-API direct path over pure OV: about 1.180 ms/frame.
- Unary graph/request path over direct C-API: about 3.292 ms/frame.

The decode-only number uses an unbounded file queue/appsink and is not additive; it is a supply-rate sanity check. Do not subtract it mechanically from any end-to-end result.

Raw evidence:

- `current_task/p10_pure_ov_d3d11.log`
- `current_task/p10_direct_d3d11_capi.log`
- `current_task/p10_file_latency_nireq4_manual.txt`
- `current_task/p10_file_debug_ovms.log`
- `current_task/P10_ZeroCopy_Review_Results.md`
- `current_task/ArchitecuteReview.md`

## Verified Architecture Facts

- `gst_video_file_infer.pbtxt` has one `GstVideoSourceCalculator` and one `GstCapiInferenceCalculator`. It has no overlay/sink and therefore no second full-resolution source branch.
- The source calculator opens one GStreamer stream for inference-only output. A second stream exists only if `FRAME_FULL` is wired for overlay/encode.
- The Windows file pipeline is `filesrc ! decodebin ! d3d11upload ! d3d11convert ! ... ! appsink`. The explicit upload/conversion is present but not independently timed.
- `GstCapiInferenceCalculator` creates an `OVMS_InferenceRequest`, adds both inputs, binds Y/UV D3D11 textures, invokes inference, reads CPU response, and destroys the request for every frame.
- C-API supports `OVMS_InferenceRequestInputRemoveData`. Input metadata can be retained while only buffers are removed/rebound. `src/main_benchmark.cpp` uses this approach for CPU buffers.

## Primary Experiments

Perform in order. Use local file input only. Preserve matching plugin config. Run three trials of 150 frames after a 20-frame warmup; save raw logs under `current_task/`.

### E1 - Reuse a D3D11 C-API Request

Modify only the unified performance test (`src/test/mediapipe/calculators/remote_tensor_performance_test.cpp`) or a dedicated Windows test target.

1. Create one request before the frame loop.
2. Add `data/y` and `data/uv` once.
3. For each decoded texture, call `OVMS_InferenceRequestInputRemoveData` on Y and UV, then rebind both with their D3D11 remote buffer types.
4. Invoke inference, consume/delete the response, and repeat.
5. Compare wall FPS and per-frame C-API time with the current create/add/set/delete-per-frame loop.

Report whether request reuse recovers a meaningful part of the 1.180 ms/frame C-API gap. Check all C-API statuses and preserve texture lifetime through inference.

### E2 - Separate Unary Graph Lifecycle From Steady State

Measure, in one instrumentation pass or a focused test:

- graph initialization/source open time;
- first-frame time;
- steady-state frames 21-150;
- graph close/EOS time;
- end-to-end client request wall time.

Do not claim the 3.292 ms/frame graph/request gap is MediaPipe scheduler cost until startup/teardown and gRPC/client time are separately accounted for.

If a long-lived graph can be driven without source changes, run it with the same local file and compare steady-state FPS. Otherwise document why it cannot be done with the current unary graph interface.

### E3 - Bounded Decode Check

Repeat decode-only with `imp_video_decode_opts_t.sync_appsink=true` or `queue_depth=1`, then compare it with the unbounded decode-only test. This shows how much the existing 247.76 FPS decode number benefits from prebuffering.

### E4 - Upload/Conversion Timing

Instrument only enough to time the pipeline's `d3d11upload`/`d3d11convert` contribution, or provide a justified alternative that uses existing GStreamer timing/tracing. Do not infer its cost from the presence of the element alone.

## Non-Goals

- Do not alter production calculator architecture yet.
- Do not move the model into the calculator.
- Do not use camera or RTSP for these measurements.
- Do not re-run the broad P10 preview matrix.
- Do not investigate Bazel cache behavior here; see `current_task/Bazel_Test_Cache_Investigation_Prompt.md`.

## Deliverable

Append a P11 section to `current_task/P10_ZeroCopy_Review_Results.md` with trial tables, matched plugin settings, raw-log paths, and a ranked conclusion:

1. request construction/binding overhead;
2. source decode/upload/conversion;
3. graph lifecycle/MediaPipe packet scheduling;
4. client request framing.

Every attribution must state its evidence and remaining uncertainty.