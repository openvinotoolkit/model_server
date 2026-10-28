# Intel MPI calculators and tests

Intel MPI is the media boundary used by the benchmark graphs. It owns source
configuration, decode, surface/tensor lifetime, and optional D3D11, OpenCL,
or VA-API interop. It does not own model execution or OVMS request handling.

## Where calculators live

The calculator implementations are split by ownership:

| Location | Contents |
| --- | --- |
| `src/mediapipe_gst_calculators/` | Production MediaPipe/GStreamer calculators used by OVMS graphs. |
| `src/test/mediapipe/calculators/` | Benchmark and integration-only calculators and test graphs. These are not part of the production calculator API. |
| `src/mpi/` | The Intel MPI C/C++ media API and its GStreamer/platform implementation. It is a dependency of calculators, not itself a MediaPipe calculator directory. |

## Calculator roles

The calculator names below describe ownership, not a required graph shape:

| Component | Purpose | Typical platform path |
| --- | --- | --- |
| `GstVideoSourceCalculator` | Decode a file, camera, or stream and publish one video frame per packet. | D3D11 on Windows; VA-API or host memory on Linux |
| `GstOpenVinoInferenceCalculator` | Run OpenVINO inference on a MediaPipe packet. | Native VA-API on Linux or D3D11 on Windows; no host fallback |
| `GstVideoOpenVinoInferenceCalculator` | A MediaPipe calculator that owns the decode stream, pulls one surface per `Process` call, imports it into OpenVINO, and emits inference results. | Native video surfaces where supported; host fallback is a separate path |
| `GstCapiInferenceCalculator` | Run an OVMS C-API request on a packet. | VA surface or D3D11 texture buffers; no host fallback |
| `GstVideoCapiInferenceCalculator` | Own decode and submit a reusable OVMS request in one calculator. | D3D11 on Windows; VA-API on Linux |
| `GstVideoOpenVinoPipelineInferenceCalculator` | A MediaPipe control calculator that inserts an OpenVINO transform into the GStreamer decode pipeline. The transform performs inference while buffers flow through GStreamer; the calculator consumes one completion packet per frame. | Native GStreamer memory where supported |

MPI should remain usable without OVMS. A graph that only decodes, encodes, or
calls OpenVINO must not acquire the OVMS server or C-API libraries through its
calculator target.

### The two video OpenVINO calculators

These names are deliberately similar because both execute direct OpenVINO
inference, but they place the inference boundary in different systems:

| Calculator | Inference boundary | Graph behavior |
| --- | --- | --- |
| `GstVideoOpenVinoInferenceCalculator` | MediaPipe `Process` method | The calculator reads a frame from MPI, creates/imports its OpenVINO tensors, calls `infer()`, and emits the result. |
| `GstVideoOpenVinoPipelineInferenceCalculator` | GStreamer transform inserted by the calculator | The calculator configures MPI with a GStreamer filter. GStreamer applies the transform during buffer flow; the calculator reads completed frames and emits progress. |

The second implementation currently uses the native element name
`ovd3d11infer` in its Windows GStreamer filter. That is an implementation
detail of the current platform backend, not part of the calculator's portable
role; a Linux backend may register a different element or use a different
native-memory transform. No Linux transform is implemented: this calculator
remains explicitly unsupported on Linux and is not substituted for the ordinary
direct-OpenVINO calculator in the baseline suite.

## Platform parity

Windows benchmark graphs use GStreamer D3D11 memory and the server-owned
`ID3D11Device`. Linux uses the same ownership boundaries with VA-API surfaces
and the actual owning VADisplay. Host NV12 and OpenCL buffers are separate
explicit experiments, not silent native-memory fallbacks. The same benchmark
target names have Linux and Windows variants.

The portability contract is the following ownership split:

1. MPI owns media acquisition and native-surface lifetime.
2. OpenVINO calculators own direct model execution.
3. OVMS calculators own only C-API request construction and submission.
4. GStreamer pipeline calculators own only pipeline configuration and packet
	accounting; the platform transform owns buffer-time inference.
5. Each platform supplies its native-memory adapter behind these same roles.

## Production files and Linux support

All sources below are under `src/mediapipe_gst_calculators/`; all target
suffixes have prefix `//src/mediapipe_gst_calculators:`.

| Calculator | File / target suffix | Linux support and limitations |
| --- | --- | --- |
| `GstVideoSourceCalculator` | `gst_video_source_calculator.cc` / `gst_video_source_calculator` | Native VA required for inference output; optional full-resolution host/VA branch; no inference |
| `GstOpenVinoInferenceCalculator` | `gst_openvino_inference_calculator.cc` / `gst_openvino_inference_calculator` | VA NV12 input packets with owning display; synchronous direct OpenVINO |
| `GstVideoOpenVinoInferenceCalculator` | `gst_video_openvino_inference_calculator_linux.cc` / `gst_video_openvino_inference_calculator` | Owns MPI VA decode and direct inference in `Process`; Windows selects the existing non-`_linux` source |
| `GstVideoOpenVinoPipelineInferenceCalculator` | `gst_video_openvino_pipeline_inference_calculator.cc` / `gst_video_openvino_pipeline_inference_calculator` | Unsupported: no Linux GStreamer inference transform; Windows currently uses `ovd3d11infer` |
| `GstCapiInferenceCalculator` | `gst_capi_inference_calculator.cc` / `gst_capi_inference_calculator` | VA C-API buffer types and shared server/decode display; optional detection boxes |
| `GstVideoCapiInferenceCalculator` | `gst_video_capi_inference_calculator.cc` / `gst_video_capi_inference_calculator` | MPI VA decode and reusable C-API request in one calculator; same shared-display requirement |

Windows input-based calculators require D3D11 textures, array slice zero,
and a common server/decode device for C-API. Linux calculators do not acquire
OVMS dependencies unless they submit C-API requests. Direct OpenVINO shares an
internal adapter in `gst_vaapi_inference.hpp`; it has no OVMS headers.

The packet holds its MPI tensor owner, surface ID and display. The inference
calculator retains the last owner until its native context is released. VA
surface ID zero is valid; `UINT32_MAX` is the invalid sentinel.

MPI API/backend locations are `intel_mpi.h`, `intel_mpi.cpp`,
`imp_mpi_impl.h`, and Linux `gst_loader.cpp` / `gst_loader.h`. Target:
`//src/mpi:intel_mpi`. Use `use_va_surface_memory=true` with
`imp_tensor_get_va_surface()` to obtain both surface and owning display.
Native output capability is detected from loaded extraction APIs even without
an injected server display. `imp_video_set_va_display()` optionally shares an
externally owned display with decode; resetting it does not disable independent
native VA support. Windows uses `use_d3d11_surface_memory=true` and
`imp_tensor_get_d3d11_texture()` instead.

Test-only `VideoDecodeInferCalculator` and `VideoDecodeInferVaCalculator` live
in `src/test/mediapipe/calculators/video_decode_infer_calculator.cc` and
`video_decode_infer_va_calculator.cc`. Their respective target suffixes are
`video_decode_infer_calculator` and `video_decode_infer_va_calculator` under
`//src/test/mediapipe/calculators:`. They are historical MPI/VA proof-of-concept
graphs with independent path/model settings, not production replacements or
substitutes for the controlled benchmark matrix.

## Nine-case benchmark matrix

All targets have prefix `//src/test/mediapipe/calculators:`. All nine baseline
cases were supported on Panther Lake, with five samples each and no skips.

| ID | Case | Target suffix | Dependencies beyond gtest | Linux |
| --- | --- | --- | --- | --- |
| 1 | Pure CPU host | `openvino_inference_benchmark` | OpenVINO | Supported |
| 2 | Pure GPU host | `openvino_inference_benchmark` | OpenVINO GPU | Supported |
| 3 | Pure OpenCL remote | `openvino_inference_benchmark` | OpenVINO GPU, OpenCL | Supported |
| 4 | Pure native GPU memory | `openvino_inference_benchmark` | OpenVINO GPU, VA-API; D3D11 on Windows | Supported; explicit VA initialization skip on unsupported machines |
| 5 | MPI decode + direct OV | `mpi_openvino_inference_benchmark` | MPI/GStreamer, OpenVINO native import | Supported |
| 6 | MPI + one OV calculator | `mpi_openvino_inference_benchmark` | MPI/GStreamer, MediaPipe, OpenVINO | Supported |
| 7 | MPI + two OV calculators | `mpi_openvino_inference_benchmark` | MPI/GStreamer, MediaPipe, OpenVINO | Supported |
| 8 | MPI + one C-API calculator | `mpi_capi_inference_benchmark` | MPI/GStreamer, MediaPipe, OpenVINO, `//src:ovms_lib` | Supported |
| 9 | MPI + two C-API calculators | `mpi_capi_inference_benchmark` | MPI/GStreamer, MediaPipe, OpenVINO, `//src:ovms_lib` | Supported |

These are distinct source files/fixtures and binaries, not one executable split
by filters. The first includes no MPI/MediaPipe/OVMS headers or server setup.
The second includes no OVMS headers or C-API setup. Only the third acquires
`//src:ovms_lib`; configured dependency queries verified that boundary.

## Linux commands

Build the benchmark targets with `bash tools/build_linux_inference_benchmarks.sh`
in the configured build environment. For a two-host build, set
`LOCAL_BUILD_CONTAINER`, `BENCHMARK_WORKER`, `REMOTE_BUILD_CONTAINER`, and
`REMOTE_CHECKOUT`; keep those values in local environment configuration.

Exact commands inside either build container:

```bash
bazel build --config=mp_on //src/test/mediapipe/calculators:openvino_inference_benchmark
bazel build --config=mp_on //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
bazel build --config=mp_on //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
```

Exact per-binary run commands inside the configured GPU test container:

```bash
bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:openvino_inference_benchmark
bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
bash tools/run_linux_inference_benchmarks.sh //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
```

Omit positional targets to run the entire matrix. The helper prepares the full
fixture, then invokes `bazel test --config=mp_on --nocache_test_results
--local_test_jobs=1` with explicit runtime environment forwarding. Action-cache
reuse is not disabled. No remote-cache endpoint was active in the supplied
environment, so cross-host remote-cache hits were not verified; existing disk
cache configuration was preserved.

The runner supplies the configured OpenVINO, GStreamer, C-API, and VA runtime
paths. Verify `gst-inspect-1.0 vapostproc` before testing and keep any driver
override in local environment configuration rather than this document.

## Results and validation

Model metadata, dimensions, 20 warm-up frames, 230 measured frames, five samples,
latency hint, one stream, and OVMS pool size are centralized in
`src/test/mediapipe/calculators/performance_benchmark_config.hpp`. Samples use
one outstanding synchronous request and consume all 250 source frames to EOS.
Pure paths use inference-only timing; MPI/graph paths use matching
completion-to-completion wall timing that includes decode and scheduling.

All nine cases passed on Panther Lake. Focused tests passed for construction,
native decode/EOS, cleanup/reopen, and byte-for-byte equality between every MPI
native frame and the pure NV12 fixture. Raw output:
`logs/panther_lake_inference_benchmark_raw.log`; condensed sample/aggregate
statistics: `logs/benchmark_summary.log`. Both were copied to the local checkout.

See the [benchmark README](../test/mediapipe/calculators/README.md) for all five
FPS/ms samples, SD/CV, exact environment setup, dependency evidence, and timing
limits. Direct MPI/OpenVINO and its graphs are close within sample variation;
C-API graphs take 2.73-2.77 ms/frame versus 1.81-1.88 for direct-OV graphs.
This is not proof of a one/two-calculator ranking or a Windows/Linux hardware
comparison. Existing Windows D3D11 diagnostics remain available; Windows
compile/runtime validation was not performed and its performance experiments
were not repeated.
