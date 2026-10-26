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

## Linux commands

Build on local SPR, synchronize the writable checkout, then build the same
targets on Panther Lake:

```bash
bash tools/build_linux_inference_benchmarks.sh
```

Defaults: local container `ovms_gst_va2`, worker
`sys_k8sworker@ov-ptl-13.sclab.intel.com`, remote container
`ovms_linux_benchmark`, checkout
`/opt/home/sys_k8sworker/atobisze/model_server`, Bazel config `mp_on_py_on`.
Override using `LOCAL_BUILD_CONTAINER`, `BENCHMARK_WORKER`,
`REMOTE_BUILD_CONTAINER`, `REMOTE_CHECKOUT`, `BAZEL_BENCHMARK_CONFIG`.
Logs: `logs/build_benchmarks_local.log`, `logs/sync_benchmarks.log`, and
`logs/build_benchmarks_remote.log`. Rsync excludes `.git`, Bazel outputs,
logs, external downloads and node_modules; it does not delete remote files.

Exact commands inside either build container:

```bash
bazel build --config=mp_on_py_on //src/test/mediapipe/calculators:openvino_inference_benchmark
bazel build --config=mp_on_py_on //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
bazel build --config=mp_on_py_on //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
```

Run all cases on Panther Lake from the local host:

```bash
ssh sys_k8sworker@ov-ptl-13.sclab.intel.com \
	'docker exec -w /ovms ovms_linux_benchmark bash -lc "LIBVA_DRIVERS_PATH=/opt/ptl-va-driver bash tools/run_linux_inference_benchmarks.sh"'
```

Exact per-binary commands inside the worker container:

```bash
LIBVA_DRIVERS_PATH=/opt/ptl-va-driver bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:openvino_inference_benchmark
LIBVA_DRIVERS_PATH=/opt/ptl-va-driver bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
LIBVA_DRIVERS_PATH=/opt/ptl-va-driver bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
```

The runner prepares the complete raw NV12 fixture using GStreamer outside the
pure binary, then invokes `bazel test --config=mp_on_py_on
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

For focused lifecycle tests, first prepare inputs with the runner, then inside
the worker container:

```bash
export OVMS_REPO_ROOT=/ovms
export PATH=/opt/gstreamer/bin:$PATH
export LD_LIBRARY_PATH=/azure-sdk-install/lib:/opt/gstreamer/lib:$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/opt/gstreamer/lib/gstreamer-1.0
export LIBVA_DRIVER_NAME=iHD LIBVA_DRIVERS_PATH=/opt/ptl-va-driver
export GST_REGISTRY=/ovms/logs/benchmark_gst_registry.bin
bazel test --config=mp_on_py_on --nocache_test_results --local_test_jobs=1 \
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

## Panther Lake results (2026-10-26)

Worker `ov-ptl-13`, GPU PCI ID `0xb081`, image `ovms_gst_va2-build:latest`,
OpenVINO `2026.5.0`, GStreamer `1.26.2`, compute runtime `26.31.39395.13-0`,
libva `2.22.0.2-87~u24.04`. Source: `recording_3_raw.avi`, MJPEG 1280x720,
30 FPS, 250 frames. The NV12 fixture contains 96,768,000 bytes.

The packaged media driver `24.3.4-1018~24.04` exposes zero VA plugin features
on this GPU. Successful runs select the worker's compatible
`25.4.0-14237~24.04` iHD driver copied into the dedicated container:

```bash
# Run on the worker, without changing the host or packaged container driver.
docker exec ovms_linux_benchmark mkdir -p /opt/ptl-va-driver
docker cp /usr/lib/x86_64-linux-gnu/dri/iHD_drv_video.so ovms_linux_benchmark:/opt/ptl-va-driver/iHD_drv_video.so
```

A fresh container needs writable sources, GPU devices and the worker's group
IDs (`getent group video render`):

```bash
docker run -dit --name ovms_linux_benchmark --entrypoint bash \
	--device=/dev/dri --group-add 44 --group-add 992 \
	-v "$HOME/atobisze/model_server:/ovms" -w /ovms ovms_gst_va2-build:latest
```

The helper selects the override using `LIBVA_DRIVERS_PATH`. Verify
`gst-inspect-1.0 vapostproc` first. A future image should package a recent
Panther Lake media driver. `/azure-sdk-install/lib` must be in the C-API
runtime library path; the run helper adds it. Use the worker's proxy environment
for package access, if needed. No host drivers were modified.

Raw artifact: `logs/panther_lake_inference_benchmark_raw.log`; condensed
records: `logs/benchmark_summary.log`. Both are host-visible on the worker and
copied back locally. They are generated, gitignored artifacts. Every successful
sample reports `warmup=20 measured=230 consumed=250 eos=1`.

Five **ms/frame** sample values:

| Case | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 CPU host | 6.388495 | 6.167282 | 6.159034 | 6.157960 | 6.155765 |
| 2 GPU host | 1.215373 | 1.191301 | 1.193493 | 1.197354 | 1.198069 |
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

The new targets require the normal `win_mp_on_py_on` configuration and Windows
GStreamer/OpenVINO/OpenCL environment. The pure binary also needs a prepared
NV12 fixture. The Linux shell fixture-preparation script is not a Windows runner.
