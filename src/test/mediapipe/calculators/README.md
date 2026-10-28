# Inference benchmarks and test calculators

## Ownership and targets

Production calculators live in `src/mediapipe_gst_calculators/`. MPI's media
API and platform backends live in `src/mpi/`. This directory owns benchmark
fixtures, integration tests, and test-only `VideoDecodeInferCalculator` /
`VideoDecodeInferVaCalculator` implementations. See the
[calculator reference](../../../mpi/README.md) for individual platform limits.

Three separate sources/fixtures produce three separate binaries. The same
target names have Linux and Windows variants; platform selections choose real
implementations, not empty Windows-only translation units.

All target names below have prefix `//src/test/mediapipe/calculators:`.

| Target | Source | Dependencies |
| --- | --- | --- |
| `openvino_inference_benchmark` | `openvino_inference_benchmark.cpp` | OpenVINO, OpenCL, VA-API/D3D11, gtest; no MPI, MediaPipe, OVMS headers/server |
| `mpi_openvino_inference_benchmark` | `mpi_openvino_inference_benchmark.cpp` | MPI, GStreamer/MediaPipe, OpenVINO, interop; no OVMS headers/server |
| `mpi_capi_inference_benchmark` | `mpi_capi_inference_benchmark.cpp` | MPI, GStreamer/MediaPipe, OpenVINO, `//src:ovms_lib`; owns C-API server/settings/requests |

Configured Bazel queries verified that `//src:ovms_lib` is absent from both
OpenVINO binaries and present in C-API. Evidence is retained in
`logs/dependency_split_no_ovms.log` and `logs/dependency_split_capi.log`.

## Nine-case matrix

Every baseline case was supported and measured on Panther Lake; none was
skipped or pending in the recorded run. GPU cases do not silently fall back
to CPU inference or a different memory path.

| ID | GoogleTest case | Binary | Path | Linux |
| --- | --- | --- | --- | --- |
| 1 | `PureOpenVinoBenchmark.CpuHost` | `openvino_inference_benchmark` | Host NV12 -> direct CPU | Supported |
| 2 | `PureOpenVinoBenchmark.GpuHost` | `openvino_inference_benchmark` | Host NV12 -> direct GPU | Supported |
| 3 | `PureOpenVinoBenchmark.OpenClRemote` | `openvino_inference_benchmark` | Two OpenCL buffers -> direct GPU | Supported |
| 4 | `PureOpenVinoBenchmark.NativeGpuMemory` | `openvino_inference_benchmark` | VAContext / native NV12 surface; D3D11 on Windows | Supported; explicit skip for unavailable VA display/context initialization |
| 5 | `MpiOpenVinoBenchmark.Direct` | `mpi_openvino_inference_benchmark` | MPI decode -> native import -> direct OV | Supported |
| 6 | `MpiOpenVinoBenchmark.OneCalculator` | `mpi_openvino_inference_benchmark` | `GstVideoOpenVinoInferenceCalculator`, inference in `Process` | Supported |
| 7 | `MpiOpenVinoBenchmark.TwoCalculators` | `mpi_openvino_inference_benchmark` | `GstVideoSourceCalculator` -> `GstOpenVinoInferenceCalculator` | Supported |
| 8 | `MpiCapiBenchmark.OneCalculator` | `mpi_capi_inference_benchmark` | `GstVideoCapiInferenceCalculator` | Supported |
| 9 | `MpiCapiBenchmark.TwoCalculators` | `mpi_capi_inference_benchmark` | `GstVideoSourceCalculator` -> `GstCapiInferenceCalculator` | Supported |

Only cases 8-9 require OVMS. The production
`GstVideoOpenVinoPipelineInferenceCalculator` is distinct from case 6: it
configures a GStreamer inference transform and consumes completion/progress
packets; inference does not run in its MediaPipe `Process`. Its current Windows
transform is `ovd3d11infer`; Linux transform support remains unsupported, not a
substitute for any of these baseline cases.

## Configuration and timing

`performance_benchmark_config.hpp` centralizes model/input names, 672x384
dimensions, 20 warm-up frames, 230 measured frames, five samples,
`PERFORMANCE_HINT=LATENCY`, `NUM_STREAMS=1`, and OVMS request-pool size 4.
All paths execute synchronously with one outstanding request. The OVMS pool
does not represent four concurrent benchmark workers. No model settings change
between cases.

Set `OVMS_BENCHMARK_REPEAT_COUNT` to a positive integer to repeat the source
and scale both the warm-up and measured frame counts by that value. For example,
`OVMS_BENCHMARK_REPEAT_COUNT=5` prepares a five-copy source and runs 100
warm-up plus 1,150 measured frames. The runner generates the repeated AVI with
the same GStreamer raw-video concat and MJPEG encode path for every repeat
count, then regenerates the matching NV12 fixture. Compare only runs prepared
with this same runner path.

Each sample consumes the complete 250-frame source. Graphs omit `MAX_FRAMES`
and require EOS with exactly 250 outputs. Direct MPI also reads to EOS. Pure
OpenVINO reads the full prepared raw NV12 fixture and rejects short, partial,
or extra frames. All paths use identical pixels, model weights, and NV12 to
BGR/NCHW preprocessing. OVMS adds this preprocessing when its native device is
registered, so its fixture supplies the original IR, not a preprocessed model.

Two timing groups must be distinguished:

- Cases 1-4 sum synchronous `infer()` durations for frames 21-250. Loading,
	tensor binding and GPU upload are excluded. These are inference ceilings,
	not transfer-inclusive end-to-end measurements.
- Cases 5-9 measure wall time from completion of frame 20 to completion of
	frame 250, including decode availability, tensor binding, inference and graph
	scheduling. Observers timestamp graph output production, not delayed poller
	draining. Initialization, compilation, warm-up and final EOS/teardown are
	excluded. Decode can overlap inference.

Within each group timer boundaries match. Comparing groups does not isolate
calculator overhead. Each sample reports FPS and ms/frame; sample SD uses
denominator `n-1`, and CV is `100 * SD / mean`, separately for both metrics.

An experimental OVMS C-API path that imported both NV12 planes through one
`VAContext::create_tensor_nv12` call was evaluated separately from this matrix.
It was opt-in only and was removed because five-sample comparisons showed no
reliable performance difference from the existing independent Y/UV imports:
approximately +2.1% for one calculator and +0.5% for two calculators, within
run-to-run variation. The supported C-API path remains the independent
per-plane import, and the paired experiment must not be used to interpret the
baseline measurements.

## Linux commands

Build the benchmark targets in the project's configured Linux build container:

```bash
bash tools/build_linux_inference_benchmarks.sh
```

The build helper requires `LOCAL_BUILD_CONTAINER`, `BENCHMARK_WORKER`,
`REMOTE_BUILD_CONTAINER`, and `REMOTE_CHECKOUT` to be set for a two-host build.
Use `BAZEL_BENCHMARK_CONFIG` to select the Bazel configuration.
Logs: `logs/build_benchmarks_local.log`, `logs/sync_benchmarks.log`, and
`logs/build_benchmarks_remote.log`. Rsync excludes `.git`, Bazel outputs,
logs, external downloads and node_modules; it does not delete remote files.

Exact commands inside either build container:

```bash
bazel build --config=mp_on //src/test/mediapipe/calculators:openvino_inference_benchmark
bazel build --config=mp_on //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
bazel build --config=mp_on //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
```

Run the cases inside the configured GPU test container:

```bash
bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:openvino_inference_benchmark
bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
```

The runner prepares the complete raw NV12 fixture using GStreamer outside the
pure binary, then invokes `bazel test --config=mp_on
--nocache_test_results --local_test_jobs=1 --test_timeout=900 --test_output=all`
with explicit library/driver/fixture environment forwarding. Serial targets
avoid concurrent GPU measurements. Neither helper disables action-cache reuse.
No active remote-cache endpoint was found in the supplied environment;
cross-host remote-cache hits were not verified. Existing named disk-cache
settings are preserved. Configure an actual remote-cache endpoint separately
if cross-host action reuse is required.

Overrides: `VIDEO_PATH`, `OVMS_REPO_ROOT`, `OVMS_PERF_MODEL_XML`,
`OVMS_PERF_NV12_PATH`, `OVMS_PERF_RENDER_NODE`, `LIBVA_DRIVERS_PATH`,
`GST_PLUGIN_PATH`, `GST_REGISTRY`, `BAZEL_BENCHMARK_CONFIG`, `BENCHMARK_LOG`.
The pure binary reads the fixture without invoking GStreamer or MPI. The runner
reads dimensions from the shared configuration header.

For focused Linux runs, `BENCHMARK_TEST_FILTER` forwards a GoogleTest filter
and `OVMS_PERF_DIAGNOSTICS=stages` reports per-frame bind/upload, synchronous
`infer()` wall time, and output-access durations for the pure OpenVINO binary.
`OVMS_PERF_DIAGNOSTICS=profile` additionally enables OpenVINO `PERF_COUNT` and
reports per-node timings; profiling materially perturbs inference latency and
is diagnostic-only. The default is `off`. Stage and profile records are saved
in the raw `BENCHMARK_LOG` artifact.

For MPI-direct E2 measurements, set `OVMS_MPI_E2_DIAGNOSTICS=stages` and filter
to `MpiOpenVinoBenchmark.Direct:MpiOpenVinoDiagnostics.DecodeSupply`. The
diagnostic reports frame-read/pull, VA surface extraction, tensor import and
binding, synchronous inference, output handling, release, and completion-wall
time. Decode-supply reports app-sink pull/drain with the same full-frame/EOS
check. Linux file appsinks are currently unbounded (`max-buffers=0`), so this
is not isolated decoder latency and must not be subtracted mechanically from
the direct path. The diagnostics are MPI/OpenVINO-only and do not depend on
OVMS.

Set `OVMS_MPI_E2_SURFACE_SYNC=on` with the E2 stage flag to time an explicit
`vaSyncSurface` before direct inference. This is a diagnostic-only control for
producer readiness; it does not change the default inference path.

Set `OVMS_MPI_E2_SYNC_APPSINK=on` to enable MPI's file-mode `sync_appsink`
option (`max-buffers=1`, `drop=false`) for both direct and decode-supply E2
cases. This is a bounded-queue diagnostic control; the default remains the
existing unbounded appsink.

Set `OVMS_MPI_E2_PREINITIALIZE_OV=on` for the direct case to initialize the
VAContext, compile the model, and create the request on the same injected
VADisplay before opening the decoder. This separates compilation from decoder
startup without changing model settings. It is an E2 diagnostic control; the
default still compiles on the first decoded frame.

The Linux-only `MpiCapiBenchmark.PairedNativeSurface` test in
`mpi_capi_inference_benchmark` replays each MPI VA surface through direct OV
and the in-process OVMS C-API before releasing the frame tensor. It alternates
call order, checks equal detection counts, and reports per-path operation
timings plus ms/frame and FPS. Run it with
`BENCHMARK_TEST_FILTER=MpiCapiBenchmark.PairedNativeSurface`; the C-API does
not expose every compiled-model property, so its configuration evidence is
reported separately from direct OpenVINO runtime properties.
`MpiCapiBenchmark.CapiNativeSurfaceOnly` uses the same setup but runs only the
C-API path, avoiding direct-OV call-order interference.
Set `OVMS_CAPI_LOG_LEVEL=DEBUG` or `TRACE` to change the benchmark server log
level. DEBUG emits per-request C-API total timings and OVMS compiled plugin
properties when supported; verbose logging is diagnostic-only and can perturb
latency. The default remains INFO.

Use the Linux-only `MpiCapiBenchmark.FairFullPathComparison` for the direct-OV
versus C-API throughput comparison. Direct OV compilation and C-API model
loading finish before timing. Each pass starts its timer immediately before
decoder open and ends after all 250 frames and EOS; no warm-up frames are
excluded. Both use one synchronous request, alternate pass order per sample,
and verify matching detections. Do not compare the pure infer-only ceiling
directly against a decode-inclusive C-API path.

Set `OVMS_MPI_E4_DIAGNOSTICS=stages` to report C-API calculator `Process` wall
time and its read/pull, surface extraction, request rebinding, synchronous
`OVMS_Inference`, response access, and response destruction buckets. The
monolithic calculator includes decode pull; the split inference calculator
does not. `OVMS_MPI_E4_WARMUP_FRAMES` defaults to 20. This is opt-in timing and
does not alter normal runs.

For the monolithic one-C-API graph, set `OVMS_MPI_E4_VA_SYNC=on` to time an
explicit `vaSyncSurface` immediately before input binding. The E4 output reports
this as `surface_sync_ms_per_frame`; the default is off.

For focused lifecycle tests, first prepare inputs with the runner, then inside
the worker container:

```bash
export OVMS_REPO_ROOT=/ovms
export PATH=/opt/gstreamer/bin:$PATH
export LD_LIBRARY_PATH=/azure-sdk-install/lib:/opt/gstreamer/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/opt/gstreamer/lib/gstreamer-1.0
export LIBVA_DRIVER_NAME=iHD
export GST_REGISTRY=/ovms/logs/benchmark_gst_registry.bin
bazel test --config=mp_on --nocache_test_results --local_test_jobs=1 \
	--test_timeout=900 --test_output=all --test_env=OVMS_REPO_ROOT \
	--test_env=PATH --test_env=LD_LIBRARY_PATH --test_env=GST_PLUGIN_PATH \
	--test_env=LIBVA_DRIVER_NAME --test_env=LIBVA_DRIVERS_PATH --test_env=GST_REGISTRY \
	--test_filter='MpiOpenVinoIntegration.*' \
	//src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
```

These tests cover graph construction, native decode/EOS, teardown/reopen, strict
native-memory selection, and byte-for-byte equality of every native frame's
Y/UV pixels against the pure fixture. The benchmark graphs additionally test
inference and repeated cleanup for both calculator shapes.

## Linux benchmark results

Benchmark logs and hardware-specific results are local artifacts. Do not add
worker names, driver paths, credentials, or host-specific measurements to this
tracked document. Record those details in local task notes when needed.
| 3 OpenCL | 1.171954 | 1.155088 | 1.156161 | 1.155725 | 1.157448 |
| 4 VA native | 1.500315 | 1.503836 | 1.493484 | 1.496956 | 1.499236 |
| 5 MPI direct OV | 1.785505 | 1.828692 | 1.777307 | 1.937910 | 2.003527 |
| 6 OV one calculator | 1.877751 | 1.720124 | 1.972588 | 1.764202 | 1.726044 |
| 7 OV two calculators | 2.003299 | 1.746195 | 1.936461 | 1.975290 | 1.714607 |
| 8 C-API one calculator | 2.762103 | 2.760860 | 2.817375 | 2.755178 | 2.767134 |
| 9 C-API two calculators | 2.608935 | 2.820721 | 2.757273 | 2.770186 | 2.684708 |

Five corresponding **FPS** sample values:

| Case | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 156.531385 | 162.145974 | 162.363131 | 162.391442 | 162.449338 |
| 2 | 822.792821 | 839.418166 | 837.876375 | 835.174949 | 834.676246 |
| 3 | 853.275476 | 865.734739 | 864.931402 | 865.257823 | 863.969770 |
| 4 | 666.526478 | 664.966136 | 669.575479 | 668.022489 | 667.006494 |
| 5 | 560.065584 | 546.839061 | 562.649028 | 516.019847 | 499.119862 |
| 6 | 532.551969 | 581.353528 | 506.948111 | 566.828512 | 579.359633 |
| 7 | 499.176683 | 572.673867 | 516.405856 | 506.254858 | 583.223878 |
| 8 | 362.042976 | 362.206008 | 354.940286 | 362.953004 | 361.384685 |
| 9 | 383.298227 | 354.519339 | 362.677141 | 360.986644 | 372.480052 |

Aggregates use sample standard deviation:

| Case | Mean ms | SD ms | CV ms % | Mean FPS | SD FPS | CV FPS % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.205707 | 0.102275 | 1.648072 | 161.176254 | 2.599099 | 1.612582 |
| 2 | 1.199118 | 0.009502 | 0.792430 | 833.987711 | 6.554020 | 0.785865 |
| 3 | 1.159275 | 0.007140 | 0.615929 | 862.633842 | 5.271227 | 0.611062 |
| 4 | 1.498765 | 0.003857 | 0.257349 | 667.219415 | 1.717464 | 0.257406 |
| 5 | 1.866588 | 0.099776 | 5.345349 | 536.938676 | 28.114737 | 5.236117 |
| 6 | 1.812142 | 0.109802 | 6.059254 | 553.408351 | 32.511711 | 5.874814 |
| 7 | 1.875170 | 0.134734 | 7.185138 | 535.547028 | 39.365918 | 7.350600 |
| 8 | 2.772530 | 0.025428 | 0.917123 | 360.705392 | 3.270727 | 0.906759 |
| 9 | 2.728364 | 0.082589 | 3.027054 | 366.792281 | 11.247910 | 3.066561 |

Comparison limits:

- Direct MPI/OpenVINO and its one/two-calculator graphs are close at
	1.81-1.88 ms/frame. Their 5-7% CV exceeds the mean differences: this does
	not establish a calculator-count ranking or a graph advantage over direct OV.
- C-API graphs take 2.73-2.77 ms/frame versus 1.81-1.88 for OV graphs. The
	one/two C-API difference is small relative to sample variation.
- GPU host/OpenCL inference ceilings are about 1.20/1.16 ms/frame; VA is
	1.50 ms/frame. VA uses surface/image rather than buffer kernels. Upload is
	excluded, so these are not end-to-end transfer or zero-copy speedup claims.
- Pure VA versus MPI/VA (1.50 versus 1.87 ms/frame) shows an additional
	decode/import/scheduling envelope, not an independently isolated decode cost.
- GPU power/clocks and shared-worker activity were not pinned. Windows D3D11
	performance experiments were not repeated. Meteor Lake versus Panther Lake
	hardware would confound cross-machine Windows/Linux comparisons.

## Windows compatibility

The same three target names have host/OpenCL/D3D11 implementations on Windows.
Windows compile/runtime validation remains outstanding in this Linux session.
The legacy Windows diagnostic fixture and runner remain available:

```powershell
.\tools\run_remote_tensor_performance.ps1 -BuildOnly
.\tools\run_remote_tensor_performance.ps1 -NoTestCache -BenchmarkTrialCount 5 -GTestFilter '*Performance'
```

The new targets require the normal `win_mp_on` configuration and Windows
GStreamer/OpenVINO/OpenCL environment. The pure binary also needs a prepared
NV12 fixture. The Linux shell fixture-preparation script is not a Windows runner.
