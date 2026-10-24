# Fixed parser package versus local "Stock" OVMS

Read-only comparison, 2026-10-24. Reference installation: `C:\llm\ovms`. Fixed inputs: [LOCAL-INPUTS.lock.json](testing/fixed-package-20261024/LOCAL-INPUTS.lock.json). Full file identities and import comparison: [STOCK-COMPARISON.json](testing/fixed-package-20261024/STOCK-COMPARISON.json).

**The local reference is a stock-version EXE with a modified runtime installation, not a pristine vendor package.** No file in that installation, no SDK DLL, and no fixed-lock entry was changed by this comparison. Only `--version`, file reads/hashes and static PE inspection were used; no model was started.

## Version and build profile

`C:\llm\ovms\ovms.exe --version` returned exit 0 and:

```text
OpenVINO Model Server 2026.4.0.530dc63f
OpenVINO backend 2026.4.0-22930-61afcb26271-releases/2026/4
OpenVINO GenAI backend 2026.4.0.0-3401-5f7f1278107
Bazel build flags: --config=win_mp_on_py_on
```

Our parser EXE is the pinned `5af105a04` product-source build, frozen with documentation at `eb76a74f7`, profile `win_mp_on_py_off`. Its helper Python runtime is retained by the fixed packaging plan to reproduce the original launcher environment; that does not enable Python calculators.

| Property | Fixed parser build | Local Stock installation |
| --- | --- | --- |
| EXE bytes | 22,596,608 | 22,731,784 |
| EXE SHA256 | `f472d619097b3d64c3899a4628f219381941e99bb388d9c9c8a1d84956475397` | `073ed783610883f9060fe861736c683a7edff59937b25dbd18e1fdeea14007fc` |
| Python feature profile | `py_off` | `py_on` |
| Direct EXE imports | No `python312.dll` | Includes `python312.dll`; otherwise the direct import-name sets match |
| GPU DLL | Stock isolated SDK RC1 | Previously recorded custom GPU reference |

## SHA256 comparison: 29 EXE/DLL entries

**21 identical, 5 different, 3 missing from the reference root.** These counts cover the fixed lock's `bin/` entries, not licenses, helper Python contents or a full recursive equivalence claim.

Identical: 20 of the 21 SDK DLLs (including `openvino.dll`, `openvino_genai.dll`, `openvino_tokenizers.dll`, CPU/NPU plugins and frontends), plus `tbb12.dll`. Thus the observed differences do not indicate a different Core/GenAI/Tokenizers binary for this local reference.

The five different files:

| File | Fixed SHA256 | Local Stock SHA256 |
| --- | --- | --- |
| `ovms.exe` | `f472d619097b3d64c3899a4628f219381941e99bb388d9c9c8a1d84956475397` | `073ed783610883f9060fe861736c683a7edff59937b25dbd18e1fdeea14007fc` |
| `openvino_intel_gpu_plugin.dll` | `b90b57aecb9d8bbb7b0a1e945e45a088e94ea7baf61eaf5dd7d8fe5e06c9e55d` | `21a2fd9f5d856c297416059325832f0f902c84a4e78b840f859ca508f5aea66d` |
| `git2.dll` | `7eb7ff327997dc19db29927153d54b1261df0888378f3c5d0ddf7c72f51998dd` | `f5faca49993a93090f5d2b64db9aadc9d103f9ca30ee47e18408e3f1911de2a2` |
| `libcurl-x64.dll` | `f31fa58061e162ff4702b37224404df210346df6dff4a49e92d07b8dd5217334` | `08ca9a464f00e1ca29dd10d4770f98664b5dfc2f5f237eda942095b9546b11ef` |
| `opencv_world4140.dll` | `a192cbf006faf48c054468fa2a2e56e3f680b401c23b8f7fb24373ff75654c46` | `3c25b93a0dc1dc99f28ac57a2fee300ada5b6acae48f7cd502ffbec66b43783c` |

Different helper-DLL hashes alone do not establish custom source patches: signing, build configuration and upstream version can also change bytes. No source-to-binary explanation was established for those three helper-DLL differences.

Additional fixed-payload files absent from `C:\llm\ovms` root: `tbbbind_2_5.dll`, `tbbmalloc.dll`, `tbbmalloc_proxy.dll`. They are the conservative release-TBB selection in the fixed lock; absence from the reference root is not proof its existing workloads fail. The reference has `espeak-ng.dll` and `espeak-ng-data`, which the fixed LLM-package scope does not include. Do not advertise equivalent speech/TTS functionality.

## The important GPU distinction

The active `C:\llm\ovms\openvino_intel_gpu_plugin.dll` hashes to `21a2fd9f...` (33,879,040 bytes). The C07 SDK recovery report identifies that full hash as the protected **custom** GPU reference:

- `C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C07\reports\REPORT.md`
- `C:\gm\qwenmonster\semantic-refit-2026.4-sdk\SDK-MANIFEST.json`

Current backup `C:\llm\ovms\plugin-backups\openvino_intel_gpu_plugin.dll.stock-20261015074213` hashes to `b90b57ae...` (34,629,896 bytes), **exactly matching the stock SDK GPU DLL in our fixed lock**. The backup's name is not the basis of that identity: the bytes match by SHA256. The date encoded in its name is not an independently verified modification time.

Therefore these are two distinct tuples:

1. Stock-version OVMS EXE `530dc63f` + installed custom GPU `21a2fd9f...` + Python-enabled runtime.
2. Our parser-retainer EXE `f472d619...` + isolated stock RC1 GPU `b90b57ae...` + `py_off` feature profile and separately retained helper Python.

This comparison confirms the installed custom GPU identity, not a fresh proof of its exact source patch/build lineage or current runtime behavior. It does not involve C6/2026.5 latest-ref resolution. Do not copy the active reference GPU into the fixed package as an incidental packaging step: that would create a third candidate needing a new lock and its own evidence.

## Python footprint and packaging implications

The `python.exe` files in the two environments are byte-identical: SHA256 `4d6f5f81a4bca11191c4c7c6b43632694d0a4ce74e068619d8fdc161d469859a`. Their directory contents are not equivalent:

| Directory | Files excluding Python caches | Bytes |
| --- | --- | --- |
| `C:\llm\ovms\python` | 35,971 | 1,484,062,436 |
| Fixed helper `...\semantic-refit-2026.4-tools\py-nupkg\tools` | 1,323 | 39,284,294 |

Counts exclude `__pycache__`, `.pyc`, `.pyo` consistently. The reference Python tree was counted, not fully fingerprinted or classified by installed package. Do not infer that all of its additional packages are required for the selected fixed `py_off` model profiles.

Use the [fixed package runbook](FIXED-WINDOWS-PACKAGE-20261024.md) unchanged for its own pinned files. Preserve this comparison as evidence of why a blind merge of `C:\llm\ovms` and `bazel-bin\src` would be incorrect. Replacing helper libraries, GPU DLL or feature profile requires an explicit new candidate; neither `--version` nor identical runtime version strings establish equivalent behavior. Package assembly, inference comparisons and loaded-module attestation remain NOT RUN in this audit.
