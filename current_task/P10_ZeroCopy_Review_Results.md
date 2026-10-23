# Video Inference Zero-Copy Review and Phase 2 Results

Date: 2026-10-21

## Executive Summary

The active C-API path is architecturally sound at the inference input boundary: the GST calculator binds VA/D3D11 native resources with remote `OVMS_BufferType` values, and OVMS creates context-backed remote tensors. The output path is CPU-owned, so the whole pipeline is not zero-copy. The current Windows local-file pipeline also includes an explicit `d3d11upload` after `decodebin`.

The user-provided local-file OVMS rerun averaged **86.26 frames/s** (75.41-94.42) over three 150-frame trials. The standalone OpenCL inference-only ceiling averaged **254.71 frames/s**, giving a raw ratio of **33.87%**. Including OpenCL input writes, standalone averaged **230.27 frames/s**, giving **37.46%**. The same D3D11 file surfaces were then measured through pure OpenVINO, direct OVMS C-API, and the MediaPipe graph. Matched wall-time averages isolate approximately **1.18 ms/frame** of C-API/request-response overhead over pure OV and a further **3.29 ms/frame** in the graph/request path over direct C-API. The RTSP/camera P10 summaries are not used.

## Component Map and Verdicts

| Component | Verdict | Behavior and evidence |
| --- | --- | --- |
| Local-file graph | In place | `gst_video_infer.pbtxt` wires a video path source to C-API inference. The experiment graph [gst_video_file_infer.pbtxt](gst_video_file_infer.pbtxt) wraps that file pipeline in a bounded `OVTENSOR` request and sets the source URI to `recording_3_raw.avi`. |
| Camera/RTSP graphs | Partial | The camera and RTSP variants add a host-NV12 overlay and a sink; they are not used for this measurement. [gst_video_camera_overlay.pbtxt](../src/mediapipe_gst_calculators/gst_video_camera_overlay.pbtxt#L1-L85), [gst_video_rtsp_overlay.pbtxt](../src/mediapipe_gst_calculators/gst_video_rtsp_overlay.pbtxt#L1-L83) |
| Video source calculator | Partial | It chooses D3D11 on Windows or VA on Linux, requires a GPU surface route, and can emit native surfaces or a host-memory full-resolution branch. Each emitted packet retains an `imp_tensor_t` owner. [gst_video_source_calculator.cc](../src/mediapipe_gst_calculators/gst_video_source_calculator.cc#L183-L233), [gst_video_source_calculator.cc](../src/mediapipe_gst_calculators/gst_video_source_calculator.cc#L247-L280), [gst_video_frame_packet.hpp](../src/mediapipe_gst_calculators/gst_video_frame_packet.hpp#L40-L59) |
| C-API inference calculator | In place for remote input; partial end to end | It selects VA Y/UV or D3D11 texture Y/UV types, binds the same native handle for both planes, and calls synchronous `OVMS_Inference`. It allocates a request per frame. [gst_capi_inference_calculator.cc](../src/mediapipe_gst_calculators/gst_capi_inference_calculator.cc#L111-L178) |
| OVMS tensor factories | In place | The model is compiled against a VA or D3D context and the corresponding buffer types create remote tensors from those contexts. [modelinstance.cpp](../src/modelinstance.cpp#L1038-L1058), [modelinstance.cpp](../src/modelinstance.cpp#L1289-L1306), [vaapitensorfactory.cpp](../src/vaapitensorfactory.cpp#L38-L49), [d3d11tensorfactory.cpp](../src/d3d11tensorfactory.cpp#L38-L42) |
| Detection overlay | Partial | It accepts host NV12 or VA, not D3D11. Host drawing is in-place on the host frame; VA drawing maps the surface and falls back to `vaGetImage`/`vaPutImage`. VA overlay is build-gated. [gst_detection_overlay_calculator.cc](../src/mediapipe_gst_calculators/gst_detection_overlay_calculator.cc#L125-L145), [gst_detection_overlay_calculator.cc](../src/mediapipe_gst_calculators/gst_detection_overlay_calculator.cc#L179-L216), [BUILD](../src/mediapipe_gst_calculators/BUILD#L70-L88) |
| Video sink | Partial | The sink selects GPU encoding for native VA/D3D11 resources; host frames use the non-GPU path. It is absent from the measured inference-only graph. [gst_video_sink_calculator.cc](../src/mediapipe_gst_calculators/gst_video_sink_calculator.cc#L120-L140) |
| GStreamer deployment boundary | Partial | Linux resolves GStreamer dynamically; Windows links it directly. The calculators are part of the OVMS target dependency list. The source returns `Unavailable` without a GPU-surface path; this is not a complete opt-in/non-GStreamer fallback boundary. [src/mpi/BUILD](../src/mpi/BUILD#L18-L27), [gst_loader.cpp](../src/mpi/gst_loader.cpp#L234-L305), [gst_video_source_calculator.cc](../src/mediapipe_gst_calculators/gst_video_source_calculator.cc#L187-L193), [src/BUILD](../src/BUILD#L2919-L2924) |

## Zero-Copy Audit

- **Input binding:** `gst_capi_inference_calculator.cc` uses `OVMS_BUFFERTYPE_VASURFACE_Y/UV` or `OVMS_BUFFERTYPE_D3D11_TEXTURE_Y/UV`; it does not bind the `GstVideoFramePacket` as `OVMS_BUFFERTYPE_CPU`. [gst_capi_inference_calculator.cc](../src/mediapipe_gst_calculators/gst_capi_inference_calculator.cc#L111-L178)
- **C-API ownership:** `InferenceTensor::setBuffer` passes the pointer and `createCopy=false` by default. The packet owner keeps the native frame alive across the synchronous request. [inferencetensor.cpp](../src/capi_frontend/inferencetensor.cpp#L36-L46), [buffer.cpp](../src/capi_frontend/buffer.cpp#L30-L38), [gst_video_frame_packet.hpp](../src/mediapipe_gst_calculators/gst_video_frame_packet.hpp#L51-L59)
- **Remote tensor creation:** VA uses the surface ID and plane; D3D11 uses the texture handle and plane. [vaapitensorfactory.cpp](../src/vaapitensorfactory.cpp#L20-L49), [d3d11tensorfactory.cpp](../src/d3d11tensorfactory.cpp#L16-L42)
- **Response:** the calculator requests CPU output, and serialization copies the output tensor into an owned CPU buffer. Small detection results make this a reasonable boundary, but it is still a copy. [gst_capi_inference_calculator.cc](../src/mediapipe_gst_calculators/gst_capi_inference_calculator.cc#L181-L213), [serialization.cpp](../src/capi_frontend/serialization.cpp#L114-L123)
- **Windows overlay:** inference can remain D3D11 remote while the camera/RTSP overlay branch is host NV12 because those graphs set `FULL_VA_SURFACE=false`. Thus the inference input is zero-copy at the C-API boundary; the entire video pipeline is not. [gst_video_camera_overlay.pbtxt](../src/mediapipe_gst_calculators/gst_video_camera_overlay.pbtxt#L1-L12), [gst_video_rtsp_overlay.pbtxt](../src/mediapipe_gst_calculators/gst_video_rtsp_overlay.pbtxt#L1-L7)
- **File decode/upload:** the measured OVMS log reports `filesrc ! decodebin ! d3d11upload ! d3d11convert ! video/x-raw(memory:D3D11Memory) ... appsink`. This is one decode branch, but it includes a CPU-to-D3D11 upload before the zero-copy inference boundary. See [p10_file_debug_ovms.log](p10_file_debug_ovms.log) and [p10_file_latency_nireq4_trials.txt](p10_file_latency_nireq4_trials.txt).

## In-Calculator Model Trade-Off

Keep the model OVMS-served. Direct `ov::CompiledModel` loading can retain remote tensors only if it uses the same GPU context and constructs remote tensors from the native handles; otherwise a host-backed tensor or context mismatch can add copies. The OVMS route already retains model versioning, reload, configuration, and metrics. Revisit direct loading only if measurements isolate material C-API/request overhead. [modelinstance.cpp](../src/modelinstance.cpp#L1038-L1058), [d3d11tensorfactory.cpp](../src/d3d11tensorfactory.cpp#L33-L42)

## Phase 2 Measurement

**Configuration and input**

- Model: `face_detection_adas`, 672x384 NV12, local input `recording_3_raw.avi`.
- OVMS config copy: [config_p10_file_latency_nireq4.json](config_p10_file_latency_nireq4.json). Effective settings in startup log: `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, `No of InferRequests=4`.
- Graph copy: [gst_video_file_infer.pbtxt](gst_video_file_infer.pbtxt). It uses a single local-file source and inference only; no overlay, encode, camera, or RTSP.
- Runtime: OpenVINO 2026.5.0, Intel integrated GPU.

**OVMS full-pipeline result**

The user's manual rerun used a 20-frame warmup and three bounded 150-frame INFO requests:

| Trial | Wall time | Throughput |
| --- | ---: | ---: |
| 1 | 1.589 s | 94.42 frames/s |
| 2 | 1.989 s | 75.41 frames/s |
| 3 | 1.686 s | 88.95 frames/s |
| Mean | 1.755 s | 86.26 frames/s |

The manual command wrote [p10_file_latency_nireq4_manual.txt](p10_file_latency_nireq4_manual.txt). Each request returned successfully after reaching `MAX_FRAMES=150`. An earlier assistant-run set averaged 69.38 FPS ([p10_file_latency_nireq4_trials.txt](p10_file_latency_nireq4_trials.txt)); it is kept as a separate exploratory sample and is not pooled with the user's rerun.

**No-MediaPipe D3D11 decode + direct C-API comparison**

The `WindowsD3D11RemoteTensorFixture.FileDecodeAndOvmsCapiInferencePerformance` test reads the same local AVI into D3D11 NV12 textures and calls the OVMS C-API directly. Its model config is pinned to `LATENCY`, one stream, and four infer requests, matching the OVMS file graph. It does not create a MediaPipe graph or calculators.

The current direct loop, like `GstCapiInferenceCalculator`, creates a request and adds/binds Y and UV input tensors for every frame. This is intentional in the current implementation, not a hidden runtime behavior. The public C API exposes `OVMS_InferenceRequestInputRemoveData`; the later P11 reusable-request experiment measured that D3D11 variant directly.

| Trial | Decode-only loop FPS | Direct decode + C-API loop FPS | C-API mean | Decode read mean during inference |
| --- | ---: | ---: | ---: | ---: |
| 1 | 218.47 | 112.26 | 7.727 ms | 1.176 ms |
| 2 | 268.80 | 119.72 | 7.294 ms | 1.053 ms |
| 3 | 262.42 | 125.67 | 6.996 ms | 0.956 ms |
| Mean | 249.90 | 119.22 | 7.339 ms | 1.062 ms |

Raw output: [p10_direct_d3d11_capi.log](p10_direct_d3d11_capi.log). This is a more useful discriminator than the OpenCL-only ceiling for the MediaPipe question: direct D3D11 + C-API is **72.36%** of the OVMS graph's rate, with an average difference of **3.20 ms/frame**. The direct test times decode and C-API in the loop but excludes graph initialization, packet scheduling, request framing/client round trip, and graph teardown. Therefore the evidence points to overhead around the MediaPipe graph/trigger lifecycle or measurement boundary; it does not prove the calculator scheduler alone is responsible.

**Pure OpenVINO D3D11 remote-tensor comparison**

The `WindowsD3D11RemoteTensorFixture.FileDecodeAndOpenVinoD3D11RemoteTensorInferencePerformance` test uses the same decoder-owned `ID3D11Device`, compiles the original detector with NV12 surface preprocessing on an `ov::intel_gpu::ocl::D3DContext`, imports the Y and UV planes from each decoded texture, calls `InferRequest::infer`, and reads detection results. It pins `LATENCY`, one stream, and four infer-request objects.

| Trial | Pure OV D3D11 loop FPS | Pure OV inference mean |
| --- | ---: | ---: |
| 1 | 126.79 | 6.233 ms |
| 2 | 153.06 | 5.339 ms |
| 3 | 137.79 | 5.810 ms |
| Mean inference stage | - | 5.794 ms |

Raw output: [p10_pure_ov_d3d11.log](p10_pure_ov_d3d11.log). For a matched comparison, use the average of the per-trial wall times rather than averaging FPS:

| Layer | Mean wall time | Equivalent FPS | Increment over previous layer |
| --- | ---: | ---: | ---: |
| D3D11 decode only | 4.036 ms/frame | 247.76 | - |
| Pure OV D3D11 remote tensors | 7.226 ms/frame | 138.39 | - |
| Direct D3D11 + OVMS C-API | 8.406 ms/frame | 118.96 | +1.180 ms/frame |
| OVMS MediaPipe graph request | 11.698 ms/frame | 85.48 | +3.292 ms/frame |

This is the current narrowest diagnosis. The pure-OV and direct-C-API loops use the same local decoder and D3D11 texture import. The C-API cost is therefore material but smaller than the remaining graph/request-path cost. The final 3.292 ms/frame includes MediaPipe graph work, calculator packet handling, graph/source lifecycle for each unary request, and client/server request framing; it does not establish that calculator scheduling alone is responsible. The current graph is a unary trigger that opens and drains a finite source on every request, so a long-lived graph is the next discriminating comparison.

The inference-only graph has one `GstVideoSourceCalculator` stream and one GStreamer pipeline for the unary request. Its source and C-API calculators are separate MediaPipe nodes, but that split does not itself duplicate video decoding. The source calculator opens a single inference stream when it has only `FRAME` or `FRAME_INFER`; a second stream is opened only when the graph also wires `FRAME_FULL` for overlay/encode work.

**Per-stage diagnostic**

A separate DEBUG run processed 150 frames at 51.09 frames/s. DEBUG logging materially reduced throughput compared with INFO, so these stage numbers are diagnostic and are not substituted for the primary throughput result. Last-150-request distributions:

| Stage | Mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| C-API request wait | 0.007 ms | 0.007 ms | 0.009 ms | 0.011 ms |
| Deserialization | 0.621 ms | 0.604 ms | 0.821 ms | 1.017 ms |
| Prediction | 7.002 ms | 6.856 ms | 7.659 ms | 14.881 ms |
| Serialization | 0.569 ms | 0.542 ms | 0.791 ms | 0.963 ms |
| Total C-API request | 12.269 ms | 11.821 ms | 13.508 ms | 43.948 ms |

See [p10_file_debug_ovms.log](p10_file_debug_ovms.log). The source reports one file decode branch. Request wait is negligible in this single-graph run, and the C-API prediction stage is the largest measured inference substage. The `d3d11upload` is the clearest remaining copy candidate; its standalone duration was not instrumented, so this run does not prove it is the dominant wall-time component.

**Standalone ceiling and percentage**

The Windows aggregate `ovms_test` does not link `openvino_remote_tensors_tests`: `src/BUILD` includes that library only for the default configuration, while the Windows branch adds only `gtest_main`. This is why invoking `ovms_test.exe` with the initial filter returned zero tests. The OpenCL ceiling is now part of the unified Windows performance gtest [remote_tensor_performance_test.cpp](../src/test/mediapipe/calculators/remote_tensor_performance_test.cpp).

The test decodes the same local AVI to 672x384 host NV12, converts frames to BGR NCHW before timing, and uploads them to FP32 OpenCL remote tensors. It uses the original `face_detection_adas` weights, `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, and four synchronous infer-request objects. The inference-only rate excludes decode, CPU color conversion, and OpenCL input upload. Transfer-inclusive timing includes the blocking OpenCL buffer writes plus inference, but still excludes decode and color conversion.

| Trial | Inference-only FPS | Input-transfer-inclusive FPS | Inference p50 | Inference p90 | Inference p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 265.733 | 241.276 | 3.748 ms | 4.070 ms | 4.572 ms |
| 2 | 259.210 | 235.179 | 3.827 ms | 4.185 ms | 4.495 ms |
| 3 | 239.187 | 214.362 | 4.053 ms | 4.393 ms | 4.976 ms |
| Mean | 254.710 | 230.272 | 3.876 ms | 4.216 ms | 4.681 ms |

The user's raw manual trial values are captured in [p10_manual_run_metrics.txt](p10_manual_run_metrics.txt); the earlier direct gtest output remains in [p10_standalone_ceiling.log](p10_standalone_ceiling.log). Both focused test executions passed.

Against the user's OVMS mean of 86.260 FPS:

- **33.87%** of the standalone inference-only ceiling: `86.260 / 254.710 * 100`.
- **37.46%** of the standalone input-transfer-inclusive rate: `86.260 / 230.272 * 100`.
- Per-trial inference-only ratios are 35.53%, 29.09%, and 37.19%; the second OVMS run is notably slower than the other two.

These are useful compute-ceiling comparisons, not identical-path efficiency. Standalone uses an FP32 BGR OpenCL input after CPU preprocessing; OVMS uses a D3D11 NV12 texture and its GPU-side model preprocessing. The plugin hint, stream count, detector weights, target iGPU, and request-object count are aligned. The four standalone requests are used sequentially, so this measures single-stream service rather than four-way concurrent throughput. The transfer-inclusive standalone metric adds only OpenCL buffer writes; it still excludes video decode and CPU color conversion.

**Interpretation of the gap**

The headline 33.87% is low because it compares OVMS end-to-end throughput to an inference-only number that excludes decode and color preprocessing. Even versus the transfer-inclusive standalone number, OVMS is 7.25 ms/frame slower on average (11.59 ms/frame versus 4.34 ms/frame). The D3D11-specific layering shows that about 1.18 ms/frame is direct C-API/request-response overhead over pure OV and about 3.29 ms/frame remains between direct C-API and the unary MediaPipe graph. This remaining delta is consistent with graph/request lifecycle or instrumentation overhead, but the current test does not separate those components.

The GStreamer source constructs a file pipeline with `filesrc ! decodebin ! d3d11upload ! d3d11convert` before the NV12 appsink [intel_mpi.cpp](../src/mpi/intel_mpi.cpp#L431-L479). This confirms an explicit D3D11 upload/conversion stage exists, but does not establish its per-frame cost or whether it dominates. One file inference branch is used, so duplicate decode is not the cause here. A separate earlier DEBUG diagnostic showed request wait p50 0.007 ms and C-API prediction p50 6.856 ms; it is from a different run and is supporting context, not a decomposition of these manual trials. The run-to-run OVMS range (75.41-94.42 FPS) is wide enough that uncontrolled GPU/background load or clock state may contribute; no utilization telemetry was recorded to identify which.

The earlier P10 RTSP/camera summary files were excluded because they use live input, which masks the decode/copy costs under review.

## History and Worktree

Recent history places the Linux VA calculator/ownership work in `0f9dafe34` (2026-09-30), the Windows D3D11 inference path in `d00160674` (2026-10-09), and current camera/MediaMTX graphs in `8a9c4cbb6` (2026-10-20). The P10 prompt/config/RTSP graph have no tracked history in this worktree, so they were treated as hints and checked against source. Pre-existing untracked model fixtures remain under `src/test/face_detection_adas/` and `src/test/face_detection_adas_nv12/`. The temporary source changes are limited to the dedicated Windows ceiling test and its BUILD target. No OVMS binary was rebuilt, no commit or push was made, and the OVMS server was stopped.

## Follow-Up

The pure-OV, direct C-API, and in-process two-calculator comparisons are complete. The remaining lifecycle work is to timestamp OVMS graph-executor checkout/reinitialization and the external gRPC client boundary in the same run. A GPU timeline is also needed to split D3D11 surface synchronization/import from preprocessing work.

## P11 Narrow Video Inference Investigation

### Reproduce the Performance Ladder

All local performance cases now live in one Windows gtest source and executable: [remote_tensor_performance_test.cpp](../src/test/mediapipe/calculators/remote_tensor_performance_test.cpp), target `//src/test/mediapipe/calculators:remote_tensor_performance_test`. The commands below use the verified repository fixture. For copied assets, replace `OVMS_PERF_MODEL_DIR`, `OVMS_PERF_MODEL_XML`, and `VIDEO_PATH`; the model directory must retain the normal OVMS layout: `<model-dir>\1\face-detection-adas-0001.xml` and its matching `.bin`.

Build the target using the normal Windows configuration, then run only the ladder cases:

```powershell
$env:PATH = 'C:\opt\openvino\runtime\bin\intel64\Release;C:\opt\openvino\runtime\3rdparty\tbb\bin;C:\Program Files\gstreamer\1.0\msvc_x86_64\bin;' + $env:PATH
$env:OVMS_REPO_ROOT = 'C:\git\model_server'
$env:OVMS_PERF_MODEL_DIR = 'C:\git\model_server\src\test\face_detection_adas'
$env:OVMS_PERF_MODEL_XML = 'C:\git\model_server\src\test\face_detection_adas\1\face-detection-adas-0001.xml'
$env:VIDEO_PATH = 'C:\git\model_server\recording_3_raw.avi'
& .\bazel-bin\src\test\mediapipe\calculators\remote_tensor_performance_test.exe `
	--gtest_filter='*Performance*'
```

The filter runs the OpenCL compute ceiling, host-NV12 GPU-preprocessing control, pure D3D11 remote-tensor loop, D3D11 decode plus OVMS C-API loop, and the in-process two-calculator graph. It excludes camera and RTSP cases. `OVMS_PERF_MODEL_DIR` is used by the temporary OVMS C-API config; `OVMS_PERF_MODEL_XML` is used by direct OpenVINO cases and normally points inside that directory.

### Baseline Ladder: What Each FPS Includes

The measurements below answer different questions. They are ordered from GPU compute alone to the complete unary OVMS request. Do **not** read the rows as an additive decomposition: decode and GPU work overlap, and some rows intentionally exclude setup work.

| Step | Measured path | FPS | What is timed | What is deliberately outside the timer | Why it matters |
| --- | --- | ---: | --- | --- | --- |
| 1 | Plain OV GPU model compute, FP32 BGR input | 254.71 | `InferRequest::infer()` on the original detector | Decode, CPU NV12-to-BGR conversion, and OpenCL buffer write | Compute ceiling only. Despite its former shorthand, this uses an **OpenCL remote buffer**; it is not a host-tensor/no-remote baseline. |
| 2 | Plain OV GPU model + OpenCL input write | 230.27 | Blocking OpenCL write plus `infer()` | Decode and CPU NV12-to-BGR conversion | Adds only the already-prepared FP32 input transfer. |
| 3 | Plain OV, host NV12 tensors + GPU preprocessing on the same D3D11 context | 237.12 | Host Y/UV tensor bind plus `infer()`; mean 4.217 ms | Decode, host-frame copy performed before timing, and output parsing | New control: same device/context, weights, NV12-to-BGR preprocessing, hints, and request count as Step 4; input is ordinary host memory. |
| 4 | Plain OV with D3D11 NV12 remote tensors | 172.59 | `infer()` after importing D3D11 Y/UV planes; mean 5.794 ms | Decode read, D3D11 tensor wrapping/set, and output parsing | Same inference model as Step 3, but the input is a decoder-owned D3D11 surface. |
| 5 | GPU D3D11 decode only | 247.76 | Read D3D11 NV12 frames through the file pipeline | Inference | Decode supply-rate sanity check, not an end-to-end component. |
| 6 | Decode through `intel_mpi` + plain OV D3D11 remote tensors | 138.39 | Full loop: decode read, D3D11 tensor import/set, inference, output parsing | OVMS/MediaPipe/client work | The directly comparable D3D11 end-to-end OV baseline. |
| 7 | OVMS C-API with an already decoded D3D11 texture | 136.26 | C-API request construction, Y/UV bind, inference, response/read; mean 7.339 ms | Decode read itself | C-API-only view; the decoder is active but its read time is reported separately. |
| 8 | `intel_mpi` decode + OVMS C-API D3D11 remote tensors | 118.96 | Full direct loop: decode plus complete C-API request/response | MediaPipe graph and gRPC client/server request | Adds 1.180 ms/frame over Step 6 in the matched P10 sample. |
| 9 | In-process two-calculator MediaPipe graph | 153.02 | Steady frames 21-170 through `GstVideoSourceCalculator` then `GstCapiInferenceCalculator` | External gRPC/client framing; first-frame startup is reported separately | Directly measures the current split architecture. Mean steady time is 6.535 ms/frame. |
| 10 | External unary MediaPipe graph request | 85.48 | Complete client request for 150 frames | Nothing in the request path | Includes graph/source startup, first frame, EOS/close, OVMS serving, and gRPC client/server framing. |

### Why D3D11 Is About 2x Below the 250 FPS Figure

The apparent $254.71 \rightarrow 138.39$ FPS drop is not one operation. Step 1 is a narrow GPU compute measurement: the frame has already been decoded, copied to host memory, converted from NV12 to FP32 BGR NCHW on the CPU, and is ready for the model. Its 3.876 ms mean excludes all of that preparation.

**Step 1 to Step 2:** the only new timed operation is `cl::CommandQueue::enqueueWriteBuffer`, a blocking host-to-OpenCL-GPU upload of the already-prepared FP32 BGR tensor. `set_tensor` was done once before the trial; it is not the difference.

**Step 2 to Step 3:** these are different experiments, not an overhead delta. Step 2 is detector inference plus an FP32 upload; Step 3 is D3D11 video decode only. Their FPS values are both supply/service rates and must not be subtracted.

**Step 3 to Step 4:** this is the remote-input control. Both use GPU NV12-to-BGR/layout preprocessing on the same D3D11 context. Host tensors average 4.217 ms and D3D11 surfaces average 5.794 ms, a **1.577 ms/frame** difference. This is a real surface-input-path cost in the tested configuration, but it is not proven to be a memory copy: the surface-memory model variant may synchronize/import the decoder surface and use a different GPU preprocessing path. A GPU timeline is still required to split synchronization, import, and any copy.

**Step 4 to Step 6:** Step 4 starts after one D3D11 frame is already available and stops at `infer()` return. Step 6 is the full per-frame plain-OV loop: it reads/decode-pulls a frame, creates two remote plane tensors, calls `set_tensor` twice, runs inference, reads the output, and counts detections. It contains no C-API or calculator.

**Step 6 to Step 8:** the 1.180 ms/frame matched difference is OVMS C-API request work: request construction/destruction, input metadata, Y/UV C-API binding, response creation, and C-API output access. The direct benchmark has no MediaPipe calculator; it manually calls `intel_mpi` then `OVMS_Inference`.

**Step 8 to Step 9:** the current graph has exactly one `GstVideoSourceCalculator` and one `GstCapiInferenceCalculator`, hence one source pipeline. Its measured steady-state rate is 153.02 FPS in-process. This is not slower than the separately sampled direct C-API result, so the evidence does not identify the packet handoff or two-calculator split as the source of the former 3.292 ms/frame gap; run-to-run GPU variation remains material.

**Step 9 to Step 10:** the external unary measurement is slower because it includes the entire request boundary. The in-process graph measured graph initialization at 0.515-0.874 ms, `StartRun` at 0.055-0.160 ms, first output at 43.195-633.487 ms, and final drain at 0.022-0.041 ms. The remaining external difference includes OVMS graph-executor and gRPC/client framing, but the historical external and new in-process trials were not simultaneous and must not be subtracted as a precise component cost.

### Graph Architecture Variants

The current architecture is the measured two-calculator graph: `GstVideoSourceCalculator` emits owned D3D11 frame packets to `GstCapiInferenceCalculator`. An older monolithic test calculator still exists at [video_decode_infer_va_calculator.cc](../src/test/mediapipe/calculators/video_decode_infer_va_calculator.cc); it performs decode and direct OpenVINO inference inside one calculator, but it is Linux/VA-oriented test code, not the current Windows D3D11/OVMS C-API architecture and not a valid Windows comparison without a D3D11 equivalent.

**Matched conditions and evidence**

P11 retained the local `recording_3_raw.avi` input, 672x384 NV12 surface, Intel iGPU, `face_detection_adas`, `PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, and `nireq=4`. The focused [remote_tensor_performance_test.cpp](../src/test/mediapipe/calculators/remote_tensor_performance_test.cpp) uses a 20-frame warmup followed by three 150-frame trials. It passed after each final measurement run. Raw output is retained in [p11_two_calculator_graph.log](p11_two_calculator_graph.log), [p11_host_tensor_gpu_nv12.log](p11_host_tensor_gpu_nv12.log), [p11_e1_request_reuse.log](p11_e1_request_reuse.log), [p11_e1_e3_direct_bounded.log](p11_e1_e3_direct_bounded.log), and [p11_e4_gst_element_latency.log](p11_e4_gst_element_latency.log).

### Secondary P11 Diagnostics

The following tests are supporting evidence for the baseline ladder above. They are not additional primary baselines.

**E1: reusable D3D11 C-API request**

The test now creates one request per trial, adds `data/y` and `data/uv` once, binds the first decoded texture directly, then removes and rebinds both buffers for every later frame. It checks every C-API status and retains the decoded tensor through synchronous inference. The first bind must not remove data because the C API correctly returns `NONEXISTENT_BUFFER_FOR_REMOVAL` for metadata-only inputs.

| Mode | Mean loop wall time | Equivalent FPS | Mean C-API time |
| --- | ---: | ---: | ---: |
| New request/input metadata every frame | 6.433 ms/frame | 155.44 | 5.838 ms |
| Reuse request; remove/rebind D3D11 buffers | 6.117 ms/frame | 163.49 | 5.580 ms |

Reusing the request recovered **0.258 ms/frame** of direct C-API time in the dedicated run, approximately 22% of the earlier 1.180 ms/frame direct-C-API-over-pure-OV gap. This identifies request construction/input metadata as a real but minority contributor. The repeat combined run has a noisy reuse third trial (6.672 ms C-API), so this is evidence of a small effect, not a stable production-performance claim.

**E3: bounded decode check**

`sync_appsink=true` changes the file pipeline to one queue buffer and one appsink buffer. The direct decode-only loop produced these wall-time averages:

| Queue mode | Mean wall time | Equivalent FPS |
| --- | ---: | ---: |
| Existing unbounded queue/appsink | 2.861 ms/frame | 349.59 |
| `sync_appsink=true` bounded queue/appsink | 2.763 ms/frame | 361.93 |

The bounded run was slightly faster, within normal run-to-run variation. The prior 247.76 FPS decode-only number therefore cannot be attributed to unbounded prebuffering alone, and it remains a supply-rate measurement rather than an additive end-to-end component.

**E4: `d3d11upload` and `d3d11convert` tracing**

GStreamer's element-latency tracer (`GST_TRACERS=latency(flags=element)`) measured the actual elements in the Windows file pipeline. It collected 170 samples per element, including startup samples:

| Element | Mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| `d3d11upload` | 0.417 ms | 0.319 ms | 0.402 ms | 2.038 ms |
| `d3d11convert` | 0.160 ms | 0.149 ms | 0.193 ms | 0.429 ms |
| Consecutive upload + convert pair | 0.578 ms | 0.472 ms | 0.576 ms | 2.418 ms |

The tracer observes element processing time, not a serialized wall-time partition: decode, GPU scheduling, and queueing overlap. It nevertheless rules out a claim that the two elements are free, while also showing their median contribution is well below the 3.292 ms/frame unary graph/direct-C-API difference.

**E2: graph lifecycle and client framing boundary**

The present unary graph interface creates a graph instance for a request and the file source opens, reaches `MAX_FRAMES`/EOS, drains, then completes the client request. It cannot drive a long-lived finite-file graph without changing the source/request contract; a completed graph cannot accept another source run. The established 150-frame unary wall measurement is therefore an end-to-end client request measurement and includes graph initialization/source open, first-frame startup, steady-state packet scheduling, EOS/close, and gRPC client/server framing.

The in-process benchmark now separates graph initialization, `StartRun`, first output, steady frames, and graph drain. It does not include the OVMS graph executor or gRPC client boundary, so those remain combined in the historical external unary measurement. A next instrumentation-only pass should timestamp the graph-executor checkout/reinitialization and client call boundary in the same INFO-level run.

**Ranked conclusion**

1. **External unary lifecycle and client request framing:** largest unresolved contributor. The in-process two-calculator graph reaches 153.02 FPS steady state, so packet handoff is not currently identified as the cause of the historical external gap; graph-executor and gRPC/client costs remain combined.
2. **Source decode/upload/conversion:** measured contributor. Upload plus conversion has a 0.472 ms p50 and 0.578 ms mean, but cannot be added mechanically to other overlapping pipeline stages.
3. **Request construction/binding:** measured minor contributor. Reusing metadata and rebinding buffers saved 0.258 ms/frame in the dedicated run, with a noisy repeat.
4. **Pure inference service:** not the primary unexplained layer. Direct C-API remains close enough to pure OV that the larger remaining gap is outside D3D11 remote-tensor support; the C-API gap itself is only partly request setup.

## Current Unified Windows Benchmark Results

Date: 2026-10-23. The unified target is
`//src/test/mediapipe/calculators:remote_tensor_performance_test`. Every
performance test uses a 20-frame warm-up followed by 150 measured frames. The
runner command below forces fresh execution and emits exactly five FPS samples
per performance test; it does not use gtest repetition.

```powershell
.\tools\run_remote_tensor_performance.ps1 -NoTestCache -BenchmarkTrialCount 5 -GTestFilter *Performance
```

All 12 selected performance tests passed. The values below are FPS; each row's
mean is the arithmetic mean of its five samples. The
`OvmsCapiInferenceWithD3D11Nv12InCalculatorPerformance` row was rerun after an
earlier 42.71 FPS outlier and uses the clean five-sample rerun shown here.

### 1. Without MediaPipe

| Test Case | 1st | 2nd | 3rd | 4th | 5th | Mean FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `OpenVinoGpuPerformance.OpenClRemoteFp32TensorInferencePerformance` | 231.12 | 231.16 | 224.33 | 248.55 | 251.09 | 237.25 |
| `WindowsD3D11RemoteTensorFixture.OpenVinoHostNv12GpuPreprocessingInferencePerformance` | 228.88 | 221.34 | 236.92 | 235.45 | 234.11 | 231.34 |
| `WindowsD3D11RemoteTensorFixture.FileDecodeAndOpenVinoD3D11RemoteTensorInferencePerformance` | 142.05 | 142.82 | 139.56 | 140.35 | 141.98 | 141.35 |
| `WindowsD3D11RemoteTensorFixture.OvmsCapiInferenceWithD3D11Nv12Performance` | 142.61 | 147.03 | 148.41 | 151.19 | 145.96 | 147.04 |
| `WindowsD3D11RemoteTensorFixture.FileDecodeWithOvmsCapiInferencePerformance` | 127.35 | 132.79 | 131.99 | 132.83 | 130.49 | 131.09 |
| `WindowsD3D11RemoteTensorFixture.FileDecodeWithD3D11Nv12SurfacePerformance` | 243.29 | 358.88 | 368.20 | 354.11 | 367.90 | 338.48 |

### 2. Single Calculator

| Test Case | 1st | 2nd | 3rd | 4th | 5th | Mean FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `WindowsD3D11RemoteTensorFixture.PureOpenVinoInferenceWithD3D11Nv12InCalculatorPerformance` | 162.66 | 157.68 | 159.88 | 169.38 | 164.00 | 162.72 |
| `WindowsD3D11RemoteTensorFixture.FileDecodeWithOpenVinoSingleCalculatorPerformance` | 152.28 | 148.58 | 164.78 | 159.32 | 148.23 | 154.64 |
| `WindowsD3D11RemoteTensorFixture.OvmsCapiInferenceWithD3D11Nv12InCalculatorPerformance` | 140.32 | 144.50 | 146.40 | 146.85 | 147.11 | 145.04 |
| `WindowsD3D11RemoteTensorFixture.FileDecodeWithOvmsCapiSingleCalculatorPerformance` | 135.88 | 140.16 | 137.81 | 143.14 | 144.71 | 140.34 |

### 3. Two Calculators

| Test Case | 1st | 2nd | 3rd | 4th | 5th | Mean FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `WindowsD3D11RemoteTensorFixture.FileDecodeWithOpenVinoSourceAndInferenceCalculatorsPerformance` | 181.20 | 171.11 | 164.90 | 183.04 | 177.89 | 175.63 |
| `WindowsD3D11RemoteTensorFixture.MediaPipeFileSourceAndOvmsCapiInferenceCalculatorsPerformance` | 150.33 | 144.66 | 154.91 | 163.32 | 159.60 | 154.56 |