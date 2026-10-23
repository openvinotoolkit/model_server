# Qwen3Coder lossless arguments: testing handoff

Date: 2026-10-23. Repository: `C:\gm\qwenmonster\semantic-refit-2026.4-src`.
Branch: `semantic-refit-2026.4`. Parent revision: `ab07f58ddfdc908bff103ae543ca1ee987a0fc8f`.
The commit containing this handoff also contains the parser fix, tests and protocol update.

## Scope and behavior

Qwen and Ornith use `--tool_parser qwen3coder --reasoning_parser qwen3`.
This change preserves Qwen3Coder argument spelling:

| Schema type | Required behavior |
| --- | --- |
| `string` | Keep literal model bytes; remove only a clean wrapping quote pair; encode once as JSON. Backslash + `t` must not become a tab, and two backslashes must not collapse. |
| `integer`, `number` | Retain the JSON token, including `1.10`, `1e3`, `9007199254740993`. Both schema types map to `ParameterType::NUMBER`. |
| `array`, `object` | Compact structural whitespace while retaining every nested numeric token. |
| `boolean`, `null` | Publish JSON `true`, `false`, `null`; retain Python `True`/`False` normalization. |

Typed malformed JSON must mark `invalidTypedArgument` and publish no call or call identity.
The validated raw JSON map is cleared between calls. Hashes, paths and tags such as `#B4E8` use the string rule.

The SAX helper uses `kParseNumbersAsStringsFlag`. Its `RawNumber` calls
`RawValue(ptr, len, kNumberType)`: the bundled RapidJSON base `RawNumber` writes a string and must not be used.
The DOM contains ordering/duplicate-detection placeholders for retained JSON values; publication uses the per-parameter raw map.

The compatible `UNKNOWN` branch preserves valid numeric/container spelling too. The original parametrized fixture lacks `float_tool` and `int_tool`; `object_tool` declares `param1`, while its cases use `arg1`. Those existing cases and expectations were retained. New tests separately use real schemas through the public parser with one-byte chunks.

Gemma's implementation is separate and unchanged. Do not edit these files or reopen Gemma acceptance:

- `src/llm/io_processing/gemma4/gemma4_tool_parser.cpp`
- `src/llm/io_processing/gemma4/gemma4_tool_parser.hpp`

Hermes, Llama, Phi, Devstral, Onyx and MiniCPM remain outside the implementation scope.

## Established results

| Check | Result |
| --- | --- |
| Before fix: `*VariousArgumentTypes*` | **FAIL**, 28/31. Failures: float `1.10`, float `1e3`, object containing `1.10`. |
| `build-ovms-test.ps1` | **PASS**, exit 0 |
| After fix: `*VariousArgumentTypes*` | **PASS**, 34/34, including 11 unchanged Onyx cases |
| Entire `*Qwen3Coder*` suite | **PASS**, 81/81 |
| `build-ovms-live.ps1` | **PASS**, exit 0 |
| `git diff --check` | **PASS**; Git reports an LF/CRLF conversion warning for the protocol document |
| Protected Gemma files against parent revision | **PASS**, no diff |
| Live Qwen/Ornith inference, HTTP/SSE acceptance, full model-dependent suite | **NOT RUN** |
| Gemma acceptance | **NOT RUN**, explicitly excluded |

Both builds report the existing repeated Bazel config warning (`short_logs`, `v2`, `monolithic`).
Parser tests load tokenizer assets; no model/server was started.

Committed evidence is under `testing/qwen-lossless-20261023/` beside this handoff:
baseline and green GoogleTest XML, plus build stderr logs (trailing whitespace removed). Large trace logs remain local at
`C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C12\reports\qwen-lossless-*.log`.
The XML records the exact test names and results; broad trace logs are not required to recover the patch.

## Repeat parser checks

Run from PowerShell. The committed runner configures the DLL paths and `PYTHONHOME`, sets the repository working directory, writes XML/log evidence and returns the test exit code.

```powershell
rtk proxy powershell -NoProfile -File C:\gm\qwenmonster\semantic-refit-2026.4-src\docs\qwenmonster\testing\run-qwen-lossless-tests.ps1 -Label handoff-focused
rtk proxy powershell -NoProfile -File C:\gm\qwenmonster\semantic-refit-2026.4-src\docs\qwenmonster\testing\run-qwen-lossless-tests.ps1 -Filter '*Qwen3Coder*' -Label handoff-qwen-suite
```

Expected: 34/34 and 81/81 respectively, exit 0. Do not add the two counts: they overlap.
New tests cover actual schema types, one-byte streaming, nested numbers, literal paths/backslashes/tags, boolean/null handling, malformed typed JSON and clearing raw values between calls.

## Rebuild if code or dependencies change

OVMS was stopped and RAM reserved for these builds. Preserve that condition; build sequentially. Do not start a model as part of this handoff.

```powershell
rtk proxy powershell -NoProfile -File C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C12\reports\build-ovms-test.ps1
# Require exit 0, then repeat both parser checks above.
rtk proxy powershell -NoProfile -File C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C12\reports\build-ovms-live.ps1
# Require exit 0. This builds ovms.exe; it does not start the server.
```

Build scripts use Bazel under `semantic-refit-2026.4-tools`, output root `C:\r20264`, VS `C:\BuildTools`, VC `14.44.35207`, and the `openvino_genai_windows_2026.4.0.0rc1_x86_64` SDK. The committed runner uses the same local SDK/tools plus OpenCV, curl and Git DLL directories. If those installations move, update the runner environment explicitly before interpreting a loader failure as a parser failure.

## Built artifact identity

These SHA256 values identify the binaries built and tested before this documentation-only handoff/commit. Rebuilding can change hashes.

| Artifact | SHA256 |
| --- | --- |
| `C:\gm\qwenmonster\semantic-refit-2026.4-src\bazel-bin\src\ovms_test.exe` | `3ac79a92468f0de2582ae99133018c1d58eaa3f0584ef9184dabc13f93857df4` |
| `C:\gm\qwenmonster\semantic-refit-2026.4-src\bazel-bin\src\ovms.exe` | `f472d619097b3d64c3899a4628f219381941e99bb388d9c9c8a1d84956475397` |

Binaries and Bazel output directories are not committed. The source commit plus existing pinned local build environment is the recovery route.

## Optional later live acceptance

This is a future task requiring explicit model-start authorization. For both Qwen and Ornith, use the selected parser flags and record the complete binary/model/config/tokenizer and loaded-DLL provenance. Capture raw HTTP response bodies and reconstructed SSE `function.arguments` strings before any client JSON reserialization. Decoding and reserializing arguments in the test client can itself erase numeric spelling.

Exercise `1.10`, `1e3`, the long integer, nested arrays/objects, quoted `C:\test`, double backslashes, and invalid typed values. Compare exact argument text for valid calls and require no published call for invalid ones. Record unary and streaming separately. Model compliance with a requested value and parser preservation are separate observations. Do not infer live model or agent readiness from the parser suite/build results.
