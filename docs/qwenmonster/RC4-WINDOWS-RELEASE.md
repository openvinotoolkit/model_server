# GEMMAMONSTER 2026.4 RC4 Windows release

Release tag: `gemmamonster-2026.4-rc4` in `DassaultFalconKing/gemmamonster_model_server_OVMS`.

This release establishes the next 2026.4 baseline. Future fixes and new builds start from this tag/main integration, not moving C6 refs, historical Bazel outputs or previous release branches. C6/2026.5, OTD and CIB2 remain separate lines; no such product changes are included.

## Composition and provenance

The parser source is the `5af105a04904713b58b13bbd82a421c6f15844fa` lossless-retention implementation. The build source checkpoint is `b7812066a`; later release commits add packaging and evidence. The official version-stamping procedure changes `src/version.hpp` during each build and restores its template afterwards. Both variants identify their Python configuration in `--version`.

The fixed runtime remains OpenVINO/GenAI 2026.4 RC1, with the measured custom GPU plugin `21a2fd9f5d856c297416059325832f0f902c84a4e78b840f859ca508f5aea66d`. The GPU is an existing protected MoE-backport binary; its exact source-to-binary build attestation has **not** been reestablished. No C6/2026.5 rebuild is substituted for it. SDK archive SHA256: `b74ee43a10bf4615b805b1bacb78f330faec9616309a86126493973ee278e1dc`.

Two packages use OVMS's Windows distribution layout: a single `ovms/` directory with EXE and DLLs, `setupvars.bat`, `setupvars.ps1`, `install_ovms_service.bat`, LICENSE and thirdparty-licenses. Python On additionally contains the compiled Python backend, `python/pyovms.pyd` and embedded CPython 3.12.10. Python Off is compiled with `PYTHON_DISABLE=1` and ships no Python runtime. Both enable MediaPipe.

The measured git2/curl/OpenCV helper cohort is retained from the preserved original build snapshot and verified against the candidate lock. A CMake rebuild regenerated git2 with a different hash; that newly generated DLL is recorded in the build snapshot but is not silently substituted into the release. The final package's eager DLL imports and runtime are checked with the retained helpers.

These are unsigned community fork packages. They are not Intel-signed official OVMS releases. Original dependency signatures, where present, are retained. A compatible Intel graphics driver and Microsoft Visual C++ 2015–2022 x64 runtime are host prerequisites; models and drivers are not bundled.

## Reproduce and package

The procedure follows this source tree's `windows_build.bat`, `windows_set_ovms_version.py`, `windows_create_package.bat` and `windows_prepare_python.bat`, corresponding to the OVMS 2026.4 Windows release flow. Path adaptation is explicit: the official scripts assume C:\opt/MSVC locations that differ on this machine. Packaging stages into a fresh directory, verifies frozen inputs, uses TLS verification and pins the Python downloads rather than reusing a mutable get-pip script.

The build wrapper is [scripts/windows/build-rc4.ps1](../../scripts/windows/build-rc4.ps1). It uses MSVC 14.44.35207 and the existing C:\r20264 Bazel cache. The actual OpenVINO/GenAI repositories resolve through C:\opt\openvino\runtime; `OpenVINO_DIR` alone does not override those WORKSPACE repositories. Verify them against the [fixed input lock](testing/fixed-package-20261024/LOCAL-INPUTS.lock.json) before rebuilding.

Preserve the original EXE/test binary and git2/curl/OpenCV DLLs under the release root's `measured-original/bin` before building. For a later reproduction the retained helpers can be recovered from the RC4 ZIP and checked against the candidate lock; do not replace them with arbitrary current PATH or newly generated copies.

```powershell
rtk proxy powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-rc4.ps1 -Python off
rtk proxy powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-rc4.ps1 -Python on
```

Run sequentially. Each build saves an immutable `build-py_off` or `build-py_on` snapshot before switching configuration. `COMPLETE.json` records source commit, flags and artifact hashes. Existing completed snapshots are never overwritten. For a later rebuild use a new output directory and release identity rather than replacing the RC4 assets.

The release fixes the distributed file bytes. Bit-for-bit reproducibility of a compiler rebuild is not claimed, and the retained custom GPU DLL remains a binary input with the source-attestation limitation stated above. Repackaging fixed inputs and compiling a successor are distinct operations.

Download CPython 3.12.10 embed-amd64 from python.org, and obtain the exact wheels listed in [PYTHON-DOWNLOADS.lock.json](testing/rc4-release/PYTHON-DOWNLOADS.lock.json). Use `pip download --only-binary=:all:` for pip 25.0.1, setuptools 80.9.0, Jinja2 3.1.6 and MarkupSafe 3.0.2. Place them in the release root's `downloads/`. The packaging wrapper checks their hashes before installing offline. Expanded stdlib and isolated Python paths follow OVMS's embedded-Python preparation protocol.

```powershell
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B scripts/windows/package-rc4.py --python off
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B scripts/windows/package-rc4.py --python on
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B scripts/windows/smoke-rc4.py --python off
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B scripts/windows/smoke-rc4.py --python on
```

Packaging verifies ZIP CRC, every extracted file hash, version/help under an isolated PATH, focused Qwen tests, and the embedded Python Tensor binding for Python On. The model smoke uses only loopback port 18137, one owned Ornith process at a time and the existing VLM/GPU profile. It requires 15463 MiB available RAM before launch and stops its own process below 1024 MiB. Existing OVMS processes prevent the test from starting.

The unit-test EXE has a Python-initializing global test environment even in the Off configuration. Focused tests therefore use the build-helper Python and repository working directory, as the original test runner does. This Python is not shipped in Off and is not added to the clean server smoke environment. The first clean-environment unit-test attempt failed before running cases for this reason; the corrected harness retains that failure log separately.

Python On enables one additional `ChatTemplateEndToEndJinjaTest.Qwen3Coder_ToolCallWithStringArgs` case in `*Qwen3Coder*`: its complete filter contains 82 cases, versus 81 in Off. It requires the four genuine OPT tokenizer/detokenizer files under `src/test/llm_testing/facebook/opt-125m`. The initial On attempt passed all 81 parser cases and failed that extra case in SetUp because these files were absent. After preparing the fixture and exposing the shipped Jinja modules to the test interpreter, the entire filter passed. This is a test-fixture/harness repair, not a product-source change.

The fixture provenance and file hashes are in [OPT-FIXTURE.lock.json](testing/rc4-release/OPT-FIXTURE.lock.json). Only the tokenizer was downloaded from `facebook/opt-125m` revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`; no OPT weights were loaded. The available standalone exporter used OpenVINO/Tokenizers 2026.3.1 and Transformers 5.0.0, as recorded in the lock. Those exporter libraries are not release runtime inputs. To reconstruct the fixture in that separate exporter environment:

```python
from pathlib import Path
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer
out = Path('src/test/llm_testing/facebook/opt-125m')
out.mkdir(parents=True, exist_ok=False)
t = AutoTokenizer.from_pretrained('facebook/opt-125m',
    revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6', trust_remote_code=False)
tokenizer, detokenizer = convert_tokenizer(t, with_detokenizer=True)
ov.save_model(tokenizer, out / 'openvino_tokenizer.xml')
ov.save_model(detokenizer, out / 'openvino_detokenizer.xml')
```

Use the recorded exporter versions and check the resulting four files against the fixture lock. The bounded export follows the tokenizer preparation used by `windows_prepare_llm_models.bat`; the broad all-model preparation script is not needed for these focused checks. The On test environment adds the extracted package's `python/Lib/site-packages` and `python` to PYTHONPATH so Jinja comes from the shipped pinned wheels.

## Parser-retainer architecture

Gemma4's local reference implementation treats native markers as structure and retains model payloads. Literal string contents are JSON-encoded once; valid JSON values use numbers-as-strings parsing and a writer whose RawNumber invokes RawValue with kNumberType. Numeric spelling is retained without a double conversion. Qwen3Coder applies that principle to XML function/parameter envelopes: keep validated raw JSON per parameter and publish it without DOM numeric rewriting. Gemma's implementation remains separate and unchanged.

“Anti-hallucination parser-retainer” describes this bounded parser responsibility: do not invent missing arguments, silently repair invalid typed values, or rewrite valid emitted payload spelling. It is not a claim that model generations cannot hallucinate or that a tool's action is automatically safe. See the [family adaptation protocol](OVMS-MODEL-FAMILY-ADAPTATION-PROTOCOL.md) for delimiters, schema policy, state/reset and completion boundaries.

## Acceptance boundaries

The original speed comparison applies to its recorded measured binaries, not automatically to these version-stamped rebuilds. RC4 package-specific receipts and per-file manifests are stored under [testing/rc4-release](testing/rc4-release). Full OVMS regression, long-context pressure, prolonged soak, agent loops and renewed Gemma parser acceptance are outside these focused release checks. Do not turn package smoke success into those acceptance claims.

The previous RC3B release and measured original EXE are retained for recovery. Local obsolete branch/build labels are maintained in the private release directory; no research branch, dirty worktree or deployment installation is deleted.
