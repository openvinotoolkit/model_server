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
| `GstOpenVinoInferenceCalculator` | Run OpenVINO inference on a MediaPipe packet. | OpenCL or host tensors |
| `GstVideoOpenVinoInferenceCalculator` | A MediaPipe calculator that owns the decode stream, pulls one surface per `Process` call, imports it into OpenVINO, and emits inference results. | Native video surfaces where supported; host fallback is a separate path |
| `GstCapiInferenceCalculator` | Run an OVMS C-API request on a packet. | Host or remote tensors, depending on the request buffer type |
| `GstVideoCapiInferenceCalculator` | Adapt a decoded video surface to an OVMS C-API request. | D3D11 on Windows; VA-API/OpenCL on Linux where supported |
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
native-memory transform.

## Platform parity

Windows benchmark graphs use GStreamer D3D11 memory and the server-owned
`ID3D11Device`. Linux equivalents should use the same calculator ownership
boundaries with VA-API surfaces where available and OpenCL or host tensors as
fallbacks. Linux test targets should be documented beside their Windows
counterparts rather than treated as an unrelated implementation.

The portability contract is the following ownership split:

1. MPI owns media acquisition and native-surface lifetime.
2. OpenVINO calculators own direct model execution.
3. OVMS calculators own only C-API request construction and submission.
4. GStreamer pipeline calculators own only pipeline configuration and packet
	accounting; the platform transform owns buffer-time inference.
5. Each platform supplies its native-memory adapter behind these same roles.
