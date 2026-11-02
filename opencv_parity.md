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

The goal is that numeric inference and OpenCV-free graphs remain usable without
OpenCV, while installing the matching optional components restores image formats
and calculator functionality on both platforms.

## Current Implementation

| Area | Linux | Windows |
| --- | --- | --- |
| Image decoder | `libovms_opencv_image_decoder.so` | `ovms_opencv_image_decoder.dll` target and loader exist |
| Decoder packaging | Staged and checked by `create_package.sh` | DLL staging and positive package smoke gate implemented; native validation pending |
| stb_image fallback | Implemented | Implemented, but packaged runtime behavior is unverified |
| OpenCV calculator plugin | Execution tests pass after applying the type-identity patch | Excluded from the build; loader reports unsupported |
| MediaPipe runtime ownership | Separate runtime shared library | Primarily in-process in `ovms.exe` |
| OpenCV-free process tests | Passing after the glog ownership fix | No equivalent process-level validation |

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

### B2: Windows Loses OpenCV MediaPipe Calculators (P1)

`third_party/mediapipe_calculators/BUILD` moves the OpenCV-dependent calculators
out of the common registration set. `src/lib/BUILD` excludes their plugin on
Windows, and `opencv_calculators_plugin_loader.cpp` returns unsupported outside
Linux. There is no Windows registration fallback.

Affected functionality includes image decoding/encoding, text and annotation
overlays, cropping, transformation, image-to-tensor conversion, segmentation
smoothing, and affine warping. The registry test explicitly removes these
calculators from Windows expectations, so that test does not protect parity.

Required decision: choose a Windows DLL boundary with explicit host/plugin API
and shared registration ownership, or an alternative implementation that retains
the required functionality without making OpenCV a startup dependency. A static
fallback can restore calculator availability temporarily, but is not full
optional-dependency parity if it forces OpenCV to load at startup.

Required work:

- Implement the chosen Windows build, loading, registration, and packaging path.
- Support actionable missing/incompatible dependency diagnostics.
- Validate calculator options, nested graphs, and packet exchange across the
  chosen boundary.
- Restore positive Windows calculator coverage instead of only erasing registry
  expectations.
- Update the currently documented Linux-only limitation when support is verified.

Status: confirmed by source inspection and documented as a current limitation.

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

The explicit test lists in `run_unit_tests.sh` and `windows_test.bat` do not include
the new dedicated calculator-plugin or OpenCV-free runtime targets. The latter
uses Linux `LD_AUDIT`; both integration targets exclude Windows.

The shared image-conversion tests allow the decoder to be unavailable, so passing
them alone does not establish that the packaged OpenCV backend works.

Required work:

- Add the focused Linux integration targets to the appropriate CI configuration.
- Add a Windows process-level equivalent that runs without OpenCV dependencies.
- Add a separate positive test that requires the decoder to load and decode an
  OpenCV-only format.
- Test clean extracted packages as well as Bazel build/runfiles layouts.

Status: confirmed by source inspection.

## Implementation Sequence

1. Linux baseline repair: B3 and B4 are resolved.
2. Windows decoder packaging and positive package test: implemented; validate B1
  on a native Windows runner before declaring it complete.
3. Decide and implement Windows calculator ownership/loading for B2. Do not copy
   the ELF unresolved-symbol strategy directly into the Windows DLL design.
4. Add the positive and negative runtime gates in B5 for both platforms.
5. Run Linux C API `mp_off` smoke coverage and the normal MediaPipe-enabled server
   checks, then validate Windows release behavior and update the documentation.

## Acceptance Criteria

- Numeric inference and an OpenCV-free MediaPipe graph work without OpenCV on
  both platforms.
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
- Neither platform has an unavoidable OpenCV startup dependency, including through
  transitively loaded runtime libraries.
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
| `//src/image_utils:no_opencv_runtime_test` | Passed after the glog ownership fix; 12 conversion tests and the server smoke checks passed |
| OpenCV-required process smoke | Passed: JPEG, OpenCV-only PFM, and numeric inference |
| Windows build and runtime/package tests | Not run; source inspection only |

The latest complete Linux run passed all 3 targets and 32 test cases. Its log is
`test_log/opencv_parity_linux_fixed.log`; the corrected calculator execution log
is `test_log/opencv_calculators_plugin_test.fixed.log`.

Next gate: run the Windows build and package scripts on a native runner, including
the new OpenCV-required package smoke test. B2 and B5 have not been implemented;
sequential work is paused at B1's native validation gate.

Run the focused Linux targets inside the selected build container, not on the
host:

```bash
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