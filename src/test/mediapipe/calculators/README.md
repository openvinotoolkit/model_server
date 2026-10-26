# Inference benchmark calculators

This directory contains controlled performance tests. Each test consumes the
same model and source video, warms up for 20 frames, then measures 230 frames.
Unless a test says otherwise, a result is reported as five independent samples
with the same device, model settings, and timer boundaries.

The benchmark test source and its current binary are:

- Source: `src/test/mediapipe/calculators/remote_tensor_performance_test.cpp`
- Bazel target: `//src/test/mediapipe/calculators:remote_tensor_performance_test`
- Windows runner: `tools/run_remote_tensor_performance.ps1`

From the repository root, run the complete performance subset with five
uncached samples:

```powershell
.\tools\run_remote_tensor_performance.ps1 -NoTestCache -BenchmarkTrialCount 5 -GTestFilter '*Performance'
```

Build without running tests:

```powershell
.\tools\run_remote_tensor_performance.ps1 -BuildOnly
```

Run one case by passing its GoogleTest name, for example:

```powershell
.\tools\run_remote_tensor_performance.ps1 -NoTestCache -BenchmarkTrialCount 5 -GTestFilter '*FileDecodeWithOpenVinoSingleCalculatorPerformance'
```

The runner supplies the repository's Windows Bazel configuration, GStreamer
and OpenVINO environment, and `OVMS_PERF_TRIAL_COUNT`. The test also accepts
`VIDEO_PATH`, `OVMS_REPO_ROOT`, `OVMS_PERF_MODEL_DIR`, and
`OVMS_PERF_MODEL_XML` for fixture overrides. The current target includes the
nine baseline paths plus diagnostics and interop tests; use a GoogleTest filter
when a narrower run is needed.

## Benchmark scope

| ID | Benchmark | Calculator or path | OVMS dependency |
| --- | --- | --- | --- |
| 1 | Pure OpenVINO CPU | Set up host inputs and infer directly with OpenVINO CPU. | None |
| 2 | Pure OpenVINO GPU | Set up inputs and infer directly with OpenVINO GPU. | None |
| 3 | Pure OpenVINO OpenCL | Create an OpenCL remote context/tensor and infer directly. | None |
| 4 | Pure OpenVINO D3D11 | Import D3D11 NV12 input and infer directly. | None |
| 5 | MPI decode + inference | Decode with MPI, then infer through the MPI/OpenVINO path. | None |
| 6 | MPI single calculator | Decode and infer inside one calculator. | None |
| 7 | MPI two calculators | `GstVideoSourceCalculator` followed by an OpenVINO inference calculator. | None |
| 8 | MPI + OVMS C-API single calculator | Decode and submit the OVMS request inside one calculator. | Required |
| 9 | MPI + OVMS C-API two calculators | Source calculator followed by a C-API inference calculator. | Required |

The current Windows implementation also contains device-identity, EOS,
camera/RTSP, host-NV12, and native `ovd3d11infer` investigations. Those are
diagnostic or interop tests, not replacements for the nine baseline cases.

## Target split

`remote_tensor_performance_test` currently combines all cases in one binary.
That is useful while developing the fixture, but it leaks `//src:ovms_lib`
into pure-OV and MPI-only tests. The intended Bazel layout is:

| Target family | Allowed dependencies |
| --- | --- |
| Pure OpenVINO | OpenVINO, OpenCL, D3D11, and test utilities only |
| MPI/OpenVINO | MPI, GStreamer/MediaPipe, OpenVINO, and platform interop only |
| OVMS C-API | MPI, GStreamer/MediaPipe, OVMS C-API, and platform interop |

The split should be made at the source-file and fixture level, not only by
adding test filters to the existing binary. A pure-OV fixture must not start an
OVMS server, and an MPI/OpenVINO fixture must not include `src/ovms.h`.

## Shared settings

Common model dimensions, model/input names, warm-up frames, measured frames,
`PERFORMANCE_HINT`, and `NUM_STREAMS` live in
`performance_benchmark_config.hpp`. Additional provider-specific options and
request count should be added there as a small value object or passed through a
test configuration object, so every benchmark variant can change them without
editing individual tests.

## Linux coverage

Every baseline row should have a Linux target or an explicit unsupported marker
in the same matrix. Linux targets should preserve the same frame counts,
sample count, timer boundaries, and model settings while substituting VA-API,
OpenCL, or host memory for D3D11. Platform-specific tests should be selected in
Bazel rather than compiled with empty Windows-only source lists.
