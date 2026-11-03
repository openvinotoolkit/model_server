# Optional OpenCV Parity Plan

## Scope

Bring the optional OpenCV functionality on `optional-opencv` to equivalent,
validated behavior on Linux and Windows. This plan records the review of branch
commit `5b366d415` against the local `main` reference on 2026-11-02.
Statuses below include subsequent working-tree fixes and their validation.

The branch separates two responsibilities:

- Encoded model-input decoding: an optional OpenCV decoder plugin, with stb_image
  fallback when the plugin or its dependencies cannot load. Resizing and precision
  conversion use OpenVINO preprocessing with either decoder.
- MediaPipe image operations: upstream OpenCV-dependent calculators moved behind
  a separate optional plugin. stb_image does not replace these calculators.

The goal is that numeric inference and graphs that do not use OpenCV calculators
remain usable without the OpenCV runtime on both platforms. Installing the
matching optional decoder and calculator plugins restores OpenCV image formats
and MediaPipe image operations.

## Current Implementation

| Area | Linux | Windows |
| --- | --- | --- |
| Image decoder | `libovms_opencv_image_decoder.so` | `ovms_opencv_image_decoder.dll` target and loader exist |
| Decoder packaging | Staged and checked by `create_package.sh` | DLL staging and positive package smoke gate implemented; native validation pending |
| stb_image fallback | Implemented | Implemented, but packaged runtime behavior is unverified |
| OpenCV calculator registration | Optional runtime plugin; execution tests pass | Optional DLL loaded into the in-process MediaPipe registries |
| OpenCV runtime dependency | Optional on server startup | Optional until an OpenCV calculator graph is deployed |
| MediaPipe runtime ownership | Separate runtime shared library | In-process in `ovms.exe`; optional calculators are a separate DLL |
| OpenCV-free process tests | Passing after the glog ownership fix | DLL loader and graph execution covered by `ovms_test`; process-level missing-DLL test pending |

Building the server successfully is not sufficient validation: the original
Linux runtime failures occurred after compilation and linking succeeded.

## Current Blockers

### B1: Windows Release Omits the Decoder DLL (Native Validation Pending)

Originally, `src/image_utils/BUILD` defined the Windows decoder and included it in
the server's optional plugin data, but `windows_create_package.bat` copied only the
OpenCV runtime DLL, not `ovms_opencv_image_decoder.dll`.

Impact: a clean Windows release package falls back to stb_image despite shipping
OpenCV. Formats supported only by OpenCV are unavailable.

Implemented:

- Stage the matching decoder DLL alongside the packaged OVMS executable.
- Fail packaging if the expected decoder artifact is missing.
- Run the existing process-level image test in a new OpenCV-required mode against
  the staged Windows package. PFM inference must succeed, proving that the decoder
  entry point loads and works instead of silently using stb_image.
- Run the child server from the package directory with only that directory and
  Windows System32 on PATH, excluding developer dependency directories.

Status: packaging changes and positive test implemented. Both positive and
OpenCV-free test modes pass on Linux. Native Windows package execution remains
required before B1 can be marked resolved; no Windows runner is available in
this session.

### B2: Windows OpenCV MediaPipe Calculators (Implementation Added; Native Validation Pending)

`third_party/mediapipe_calculators/BUILD` moves the OpenCV-dependent calculators
out of the common registration set. Windows now builds the same optional
calculator target as a DLL and loads it with `LoadLibraryEx`, installing factories
into the in-process MediaPipe registries through the exported C registration API.

This restores availability for the affected image decode/encode, text and
annotation overlays, crop/transform, image-to-tensor, segmentation smoothing,
and affine warp calculators without linking their implementations into the
Windows server executable. The decoder and calculator DLLs load OpenCV from their
own directory; deployments that do not use these features can omit the plugin
DLLs and OpenCV runtime.

Implemented:

- Build the upstream OpenCV calculator set as a Windows DLL and export its C
  registration entry point.
- Load the DLL beside OVMS or from its `lib` directory, with an environment path
  override for embedding and diagnostics when it is unavailable.
- Keep calculator option protos in the Windows host so graph config parsing uses
  the host registry while calculator factories live in the optional DLL.
- Remove the test code that erased the OpenCV calculator names from Windows
  registry expectations.
- Add a cross-platform `ovms_test` that parses calculator options and executes a
  one-pixel encoded-image graph through the loaded plugin.
- Update the MediaPipe documentation with the platform-specific dependency model.

TODO: add a package-level Windows graph smoke after the package is extracted. The
current package script checks that the calculator DLL is present, and its image
smoke validates the decoder, but neither proves that a MediaPipe graph can load
the calculator DLL from the packaged directory.

The current Windows CI build/package commands are:

```bat
windows_build.bat <short_bazel_path> --with_tests
windows_create_package.bat <short_bazel_path> [--with_python]
```

The first command runs `//src:ovms_test`, including the new option-parsing and
decoder-graph test from Bazel runfiles. The second creates `dist\windows\ovms.zip`
and runs the existing packaged decoder smoke. Extend the package stage with a
MediaPipe graph that uses `OpenCvEncodedImageToImageFrameCalculator`, then assert
the graph reaches `AVAILABLE` and returns the decoded packet from the extracted
package. That is the missing packaged-DLL check.

Status: Windows DLL build, load, registration, option-proto anchoring, packaging,
and a graph-execution gtest are implemented. Linux plugin regression tests pass.
Windows `cquery` selects `ovms_opencv_calculators.dll` and host option protos when
MediaPipe is enabled, and excludes the option protos when disabled. Native
Windows compilation, DLL loading, option parsing, graph execution, and package
validation are still required.

### B3: Linux Plugin Packet Types Do Not Match the Host (Resolved)

Two `OpenCvPluginDeployment` tests previously failed with a packet-type mismatch even though
the reported stored and requested types are both `std::string`:

- `RegistersOptionsAndExecutesDecoderGraph`
- `CtypesLocalSharedHostExecutesDecoderGraph`

Applied `third_party/mediapipe/ovms_plugin_type_identity.diff` through the MediaPipe
archive's `WORKSPACE` patch list. This switches from address-based fast type
identity to RTTI comparison across runtime libraries.

Verification:

- Both previously failing execution tests now pass.
- The locally loaded shared-host missing-OpenCV test also passes.

Status: fixed and verified in Linux container `ab56ae89a651` on 2026-11-02.
Compiler, OpenVINO, and C++ ABI settings must still match between host and plugins.

### B4: Linux Runtime Split Causes Duplicate glog Registration (Resolved)

Before the fix, `//src/image_utils:no_opencv_runtime_test` failed while starting
the server with:

```text
ERROR: something wrong with flag 'timestamp_in_logfile_name'
```

Both the executable and `libovms_mediapipe_runtime_shared.so` linked glog registration
objects. Loader tracing confirmed that the failure loaded the freshly built
MediaPipe runtime from Bazel runfiles, rather than the old staged runtime library
in the reused container.

Resolution:

- Reused `runtime_plugin_deps` to exclude the host-provided glog dependency from
  the Linux MediaPipe shared runtime. Its logging calls resolve against the
  executable's exported glog symbols instead of registering another copy.
- Kept the Windows runtime dependencies unchanged.
- Reran the OpenCV-free runtime test successfully, including server readiness,
  encoded JPEG inference, unsupported-format diagnostics, and numeric inference.

Status: fixed and verified in Linux container `ab56ae89a651` on 2026-11-02.
The original failure log is retained in `test_log/no_opencv_runtime_test.log`;
the passing result is in `test_log/no_opencv_runtime_test.fixed.log`.

### B5: Integration Coverage Is Not Gated in CI (P2)

The Linux-only dedicated calculator-plugin and OpenCV-free process targets are not
in the `run_unit_tests.sh` target list. Windows `windows_test.bat` runs
`//src:ovms_test`, which now includes the DLL registration and decoder graph test,
but there is no Windows process-level missing-plugin test. Native Windows
execution is still required.

The shared image-conversion tests allow the decoder to be unavailable, so passing
them alone does not establish that the packaged OpenCV backend works.

Required work:

- Add the focused Linux plugin and OpenCV-free integration targets to CI.
- Add a Windows process-level missing-plugin test that verifies non-OpenCV
  workloads still start and serve inference without the optional DLLs.
- Add a separate positive test that requires the decoder to load and decode an
  OpenCV-only format.
- Test clean extracted packages as well as Bazel build/runfiles layouts.

Status: confirmed by source inspection.

### Windows Decoder Performance Validation (TODO)

`//src/image_utils:image_decode_benchmark` currently compares direct OpenCV
decoding, stb decoding with BGR conversion, and the OpenCV path including the
encoded-buffer copy. It reports CSV rows with median/p95 microseconds and images
per second. This compares decoder throughput, but because the benchmark binary
links OpenCV directly it does not measure optional-DLL discovery or startup with
OpenCV libraries absent.

On a Windows build machine with the dependencies configured as in
`windows_build.bat`, build and run it against the same representative image set:

```bat
set /p bazelVersion=<.bazelversion
call C:\opt\openvino\setupvars.bat
call C:\opt\opencv_4.14.0\setup_vars_opencv4.cmd
C:\opt\bazel-%bazelVersion%-windows-x86_64.exe --output_user_root=C:\opt build --config=win_mp_on --action_env OpenVINO_DIR=C:/opt/openvino/runtime/cmake //src/image_utils:image_decode_benchmark
bazel-out\x64_windows-opt\bin\src\image_utils\image_decode_benchmark.exe src\test\images\rgb.jpg src\test\images\gray16.png
```

Save the CSV output and compare the `opencv` and `stb_bgr` rows over repeated
runs. Use identical image files, machine, and build configuration; compare the
median and p95, and confirm the reported output difference. This benchmark only
measures decoding (not image encoding).

Also verify optional runtime behavior using the packaged process smoke. The
normal `windows_create_package.bat` run exercises the DLL-present path via
`--expect-opencv`. For the DLL-absent case, copy `dist\windows\ovms` to a
temporary directory, remove `ovms_opencv_image_decoder.dll`,
`ovms_opencv_calculators.dll`, and `opencv_world*.dll`, then run:

```bat
python src\test\no_opencv_server_test.py . - C:\path\to\temporary\ovms\ovms.exe
```

The second run must start OVMS, decode JPEG through stb, return an explicit
missing-OpenCV error for PFM, and serve numeric inference. It is a functional
fallback check, not a throughput measurement.

TODO: add an optional-plugin mode to `image_decode_benchmark` that times the
actual `decodeImage(AUTO)` path, then run that mode once with the package DLLs
present and once with them removed. This is needed to measure plugin dispatch and
the true no-OpenCV runtime path rather than only direct OpenCV versus stb codecs.

### B6: Runtime Factory Archive Ordered Before Its Caller (Resolved)

Functional CI failures reported missing MediaPipe runtime symbols and timed out
waiting for graph readiness. A local branch startup reproduced the same error:
the runtime library referenced an undefined
`MediapipeFactory::MediapipeFactory(PythonBackend*)` despite compiling its object.

The custom `runtime_plugin_deps` rule flattened each root's linker inputs
separately. This placed the lazy factory archive before the C API object that
references it, so the linker skipped the implementation. Merging all roots'
`CcInfo` contexts before filtering preserves dependency order and resolves the
constructor while still excluding the host-provided glog copy.

The process regression now deploys an OpenCV-free passthrough graph and requires
its ready endpoint, inference result, and the same `AVAILABLE after handling:
ValidationPassedEvent` log used by functional tests. It passes both with OpenCV
blocked and with the OpenCV decoder installed. All 3 focused Linux targets and
32 test cases pass. The original CI image has not been rerun.

Evidence: `test_log/mediapipe_graph_runtime.fixed.log` and
`test_log/mediapipe_runtime_link_order.fixed.log`.

## Implementation Sequence

1. Linux baseline repair: B3, B4, and B6 are resolved.
2. Validate B1 and B2 with a native Windows build, graph execution, and package run.
3. Add positive and negative runtime gates in B5 on both platforms.
4. Run Linux C API `mp_off` smoke coverage and the normal MediaPipe-enabled server
   checks, then validate Windows release behavior and update the documentation.

## Acceptance Criteria

- Numeric inference and a graph without OpenCV calculators work without the
  OpenCV runtime on both platforms when the optional plugin DLLs and OpenCV
  libraries are absent.
- Encoded JPEG/PNG inputs use stb_image successfully when OpenCV is absent.
- With the matching decoder installed, a positive test proves OpenCV is selected
  and decodes an OpenCV-only format.
- Without OpenCV, an unsupported encoded format returns an actionable dependency
  error; malformed input is not silently retried after OpenCV decoding fails.
- OpenCV calculator graphs deploy and execute when optional components are
  installed, including calculator options and packet exchange with the host.
- Missing calculator dependencies fail only the dependent graph; unrelated
  graphs remain usable.
- Local/shared C API hosts and normal executables both resolve their optional
  components from the supported package layout.
- Neither platform has an unavoidable OpenCV startup dependency, including
  through transitively loaded runtime libraries.
- No duplicate glog, protobuf, or MediaPipe registrations occur at startup or
  plugin load time.
- CI exercises the positive and negative paths rather than accepting a missing
  backend as proof that the installed backend works.

## Verification Recorded So Far

Linux container: `ab56ae89a651`, Ubuntu 24.04, with OpenCV `4.14.0` and
OpenVINO/GenAI `2026.5.0.0.dev20261029`.

| Check | Result |
| --- | --- |
| `//src:ovms` build | Passed |
| C API benchmark with `mp_off` and dummy-only config | Passed |
| `//src/image_utils:kserve_image_conversion_test` | 12 tests passed |
| `//src:opencv_calculators_plugin_test` | All 8 passed after the type-identity patch |
| `WhitelistRegistered.MediapipeCalculatorsList` | Passed after making the test require the full calculator registry |
| `//src/image_utils:no_opencv_runtime_test` | Passed after the glog ownership fix; 12 conversion tests and the server smoke checks passed |
| OpenCV-required process smoke | Passed: JPEG, OpenCV-only PFM, and numeric inference |
| Windows build and runtime/package tests | Not run; Windows-configured Bazel analysis only |

The latest complete Linux run passed all 3 targets and 32 test cases. Its log is
`test_log/opencv_parity_linux_fixed.log`; the corrected calculator execution log
is `test_log/opencv_calculators_plugin_test.fixed.log`.

Next gate: run the Windows build and package scripts on a native runner, including
the new OpenCV-required package smoke test. B2 and B5 have not been implemented;
sequential work is paused at B1's native validation gate.

Run the focused Linux targets inside the selected build container, not on the
host or a release/runtime image. The following is a developer verification
reference, not an executable runtime documentation example:

```text
bazel test --config=mp_on --//:distro=ubuntu \
  --cache_test_results=no --test_output=errors --test_summary=detailed \
  //src/image_utils:kserve_image_conversion_test \
  //src:opencv_calculators_plugin_test \
  //src/image_utils:no_opencv_runtime_test
```

The earlier CI excerpt's missing ResNet model is not a parity blocker: the C API
smoke workload only uses `dummy`. The later BuildKit RPC `EOF` in that excerpt
remains a separate, undiagnosed infrastructure/build failure; no connection to
the runtime failures above has been established.