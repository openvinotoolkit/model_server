# Fixed Windows package: parser-retainer 2026.4 / RC1 runtime

Prepared 2026-10-24. **Status: local inputs inventoried and verified; package assembly, extracted-package smoke tests and package live acceptance NOT RUN.** This is a packaging runbook, not a new acceptance receipt.

The goal is to preserve the existing EXE and the selected lock's dependency bytes. Do not resolve latest refs, rebuild C6, update the SDK, or replace the selected GPU plugin ad hoc during packaging. C6 is the separate 2026.5 line. The baseline package below uses our patched OVMS with stock 2026.4 RC1 runtime files; the later custom-GPU candidate is explicitly identified by a separate lock.

## Fixed identity and evidence boundary

- Frozen source tip: `eb76a74f7aa39573d452d95443e189b1f5f4d9e8`, tag `gemmamonster-2026.4-rc3b-parsers`.
- Parser fix: `5af105a04904713b58b13bbd82a421c6f15844fa`. Product/build inputs under `src`, `third_party`, `WORKSPACE`, `versions.mk` and `.bazelrc` have no diff between these two revisions; the later changes are documentation.
- Existing `ovms.exe`: 22,596,608 bytes, SHA256 `f472d619097b3d64c3899a4628f219381941e99bb388d9c9c8a1d84956475397`, last-write time 2026-10-23 13:30:59 local.
- Existing `ovms_test.exe`: SHA256 `3ac79a92468f0de2582ae99133018c1d58eaa3f0584ef9184dabc13f93857df4`. Keep as a developer artifact; do not add it to the runtime ZIP.
- Source build profile: `win_mp_on_py_off`. The fixed package below deliberately retains the small helper Python tree used by the original launchers. This does not enable Python calculators or turn the EXE into a `py_on` build.
- Official SDK archive: `openvino_genai_windows_2026.4.0.0rc1_x86_64.zip`, SHA256 `b74ee43a10bf4615b805b1bacb78f330faec9616309a86126493973ee278e1dc`.

The older `RELEASE-CANDIDATE-20261023-RC3B-IDENTITY.json` describes an earlier EXE with SHA256 beginning `2beed15e`; do not use that binary identity for this postfix build.

The subsequent [comparison with local Stock `2026.4.0.530dc63f`](FIXED-VS-LOCAL-STOCK-20261024.md) found that `C:\llm\ovms` currently contains the custom GPU reference, whereas its stock-GPU backup matches the DLL pinned here. The fixed lock is unchanged; do not substitute that installation's active GPU or its Python-enabled EXE.

Later benchmark decision: [speed comparison and next RC composition](SPEED-COMPARISON-AND-RC-DECISION-20261024.md) selects a **separate candidate** using the custom GPU binary with our existing EXE, for reference MoE-backport continuity rather than a claimed general speed gain. The original stock-SDK lock and the baseline commands in this runbook remain unchanged. To assemble the new candidate, explicitly select `testing/fixed-package-20261024/speed-comparison-20261024/CANDIDATE-CUSTOM-GPU.lock.json` as `$lockPath`, use a new stage/archive name containing `customgpu-candidate`, and pass `--lock $lockPath` on every pre-copy and staged verification call. Neither package has extracted-package/release acceptance from this benchmark.

On 2026-10-23, parser checks passed 34/34 and 81/81, and test/server builds completed. The original [testing handoff](QWEN3CODER-LOSSLESS-TESTING-HANDOFF.md) recorded live NOT RUN at that checkpoint. Later evidence exists:

| Evidence | Established result |
| --- | --- |
| Ornith unary, 2026-10-23 13:53:28 Europe/Berlin | One `record` call, finish `tool_calls`; raw arguments retain `1.10`, `1e3`, `9007199254740993` |
| Ornith SSE, 2026-10-23 13:53:33 Europe/Berlin | Same scalar spellings and completed tool call |
| Nested JSON in that live probe | Model returned `nested:{}` and `items:[]`; it did not exercise nested-number retention |
| User-confirmed launcher use | Scripts below identify the EXE path and intended dependency environment; Ornith parameters match the live server log |
| Historical module attestation | No simultaneous EXE/DLL SHA256 snapshot of that running process was found |

Raw evidence: `C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C12\reports\live-qwen-spelling-{server.log,unary.bin,stream.bin}`. The server log includes later requests as well; the two spelling responses above have explicit timestamps. `start-ornith.ps1` was created at 14:28, after the first spelling probe; do not claim that particular file existed at 13:53. This evidence supports a working local launch, not acceptance of a redistributed ZIP or every model family. Gemma acceptance is not reopened here.

## Local paths: runtime, build tools, evidence

| Purpose | Exact local source |
| --- | --- |
| OVMS source tree | `C:\gm\qwenmonster\semantic-refit-2026.4-src` |
| Known EXE and app-local companion DLLs | `C:\gm\qwenmonster\semantic-refit-2026.4-src\bazel-bin\src` |
| Bazel real output backing that directory | `C:\r20264\i6egknn4\execroot\ovms\bazel-out\x64_windows-opt\bin\src` |
| SDK root | `C:\gm\qwenmonster\semantic-refit-2026.4-sdk\extracted\openvino_genai_windows_2026.4.0.0rc1_x86_64` |
| Core, GenAI, Tokenizers, CPU/GPU plugins and frontends | SDK root + `\runtime\bin\intel64\Release` |
| TBB release DLLs | SDK root + `\runtime\3rdparty\tbb\bin` (exclude `*_debug.dll`) |
| SDK headers/import libraries/CMake | SDK root + `\runtime\include`, `\runtime\lib\intel64\Release`, `\runtime\cmake` (build inputs, not runtime ZIP) |
| SDK archive/provenance | `C:\gm\qwenmonster\semantic-refit-2026.4-sdk\SDK-MANIFEST.json`, `dependency-resolution.json`, and the archive beside them |
| Helper Python, as used by launchers | `C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools` |
| Bazel | `C:\gm\qwenmonster\semantic-refit-2026.4-tools\bazel.exe`; output user root `C:\r20264` |
| Visual C++ toolchain | `C:\BuildTools\VC\Tools\MSVC\14.44.35207` |
| PE inspector | Toolchain + `\bin\Hostx64\x64\dumpbin.exe` |
| OpenCV distribution/licensing | `C:\opt\opencv_4.14.0`; use the actual `opencv_world4140.dll` beside the EXE for payload |
| curl distribution/licensing | `C:\opt\curl-8.22.0_2-win64-mingw`; use the actual `libcurl-x64.dll` beside the EXE for payload |
| libgit2 | `git2.dll` beside the EXE; it is not supplied by a Git CLI installation |
| Historical launch scripts | `C:\Users\testc\.omp\agent\start-ornith.ps1`, `start-gemma26.ps1` |
| Build scripts/evidence | `C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C12\reports` |

`C:\opt\openvino\runtime` is a mutable build alias/copy used by Bazel repository definitions. Its checked Core/GenAI/Tokenizers/GPU DLLs currently match the isolated SDK, but package from the explicitly pinned SDK instead. Merely passing `OpenVINO_DIR` does not replace every hard-coded Bazel repository path.

The original PATH also mentioned `C:\Program Files\Git\mingw64\bin`; that directory was absent at this inspection. It is not needed as the source of `git2.dll`, which is already app-local. Do not install a new Git distribution just to fill this historical PATH entry.

`versions.mk` names curl `8.21.0_7`, while the launch scripts name `8.22.0_2`. Current `libcurl-x64.dll` files in **both** directories and beside the EXE have the same SHA256 `f31fa58061e162ff4702b37224404df210346df6dff4a49e92d07b8dd5217334`. Folder/version labels alone therefore do not establish which download originally supplied these bytes. The lock pins the actual app-local DLL; the matching 8.22 distribution supplies captured license files.

## Machine-readable lock and read-only verifier

[LOCAL-INPUTS.lock.json](testing/fixed-package-20261024/LOCAL-INPUTS.lock.json) records 41 selected files, four directory fingerprints, source and destination paths, sizes and SHA256. It also records the SDK archive, tools, original launchers, test artifact, host system-DLL observations, and a static PE import report. Binary files are not checked into Git.

The runtime set includes all 21 SDK release DLLs, four non-debug TBB DLLs, and app-local git2/curl/OpenCV. This conservative DLL set includes additional SDK plugins; their presence is not a claim of NPU or other-family acceptance. SDK `cache.json`, model weights, model compilation caches, tests, build caches, compiler/import libraries, and unrelated host directories are excluded.

The helper Python tree is retained to reproduce the launchers' `PYTHONHOME` rather than infer it can be removed from full model operation. Its pinned inventory excludes `__pycache__`, `.pyc`, `.pyo`. The EXE's inspected direct imports do not include `python312.dll`; the stray copy beside the Bazel EXE is not separately packaged. Removing the helper Python tree later requires a separate packaging/runtime check.

The static scan of all selected `bin` PE files found **zero unresolved imports** against the planned payload, Windows API sets and this host's System32. This is not a dynamic `LoadLibrary`/GPU-driver audit. Intel GPU drivers/OpenCL, Windows system DLLs, UCRT and a compatible x64 MSVC redistributable remain host prerequisites. Do not copy DLLs out of System32. The lock's system hashes describe this host only, not mandatory Windows updates for every recipient.

Verify the frozen local inputs (does not start OVMS or a model):

```powershell
$repo = 'C:\gm\qwenmonster\semantic-refit-2026.4-src'
$python = 'C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe'
$reference = Join-Path $repo 'docs\qwenmonster\testing\fixed-package-20261024'
$lockPath = Join-Path $reference 'LOCAL-INPUTS.lock.json'
rtk proxy $python -B (Join-Path $reference 'verify-inputs.py') --lock $lockPath
if ($LASTEXITCODE -ne 0) { throw 'Frozen inputs changed; stop packaging' }
```

Expected: PASS, 51 checks, zero errors. The verifier checks bytes/directory fingerprints and the saved import report; it does not freshly run `dumpbin` or certify current host prerequisites. Do not regenerate expected hashes just to turn a mismatch green. Directory fingerprints are SHA256 of UTF-8 records sorted by relative POSIX path: `sha256<TAB>bytes<TAB>relative_path<LF>`, excluding the cache patterns above.

[INPUT-VERIFICATION.json](testing/fixed-package-20261024/INPUT-VERIFICATION.json) records the successful current-input check and two negative controls: changing the expected EXE hash is rejected, and a payload destination escaping its root is rejected. The five PowerShell blocks below were syntax-parsed without executing assembly, smoke tests or archive creation.

## Assembly recipe: reuse the frozen binaries

This recipe is intentionally **not executed** by this documentation task. It creates a new staging directory, never deletes/reuses an existing package, and verifies staged copies after copying. Run in PowerShell on this host after the input check passes. The destination is a new directory under `C:\gm\qwenmonster\packages`; keep the model directories untouched.

```powershell
$ErrorActionPreference = 'Stop'
$repo = 'C:\gm\qwenmonster\semantic-refit-2026.4-src'
$python = 'C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe'
$reference = Join-Path $repo 'docs\qwenmonster\testing\fixed-package-20261024'
$lockPath = Join-Path $reference 'LOCAL-INPUTS.lock.json'
$lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
$stage = 'C:\gm\qwenmonster\packages\rc3b-parsers-eb76a74f-fixed-20261024\ovms'
if (Test-Path -LiteralPath $stage) { throw 'Staging path already exists; choose a new name' }
rtk proxy $python -B (Join-Path $reference 'verify-inputs.py') --lock $lockPath
if ($LASTEXITCODE -ne 0) { throw 'Input verification failed' }
New-Item -ItemType Directory -Path $stage | Out-Null

function Get-PayloadPath([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative) -or $relative.Contains(':') -or
        $relative.Split('/\') -contains '..') { throw 'Unsafe destination' }
    $resolved = [IO.Path]::GetFullPath((Join-Path $stage $relative))
    $prefix = [IO.Path]::GetFullPath($stage).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Destination escapes staging root'
    }
    return $resolved
}
foreach ($entry in $lock.payload_files) {
    $target = Get-PayloadPath $entry.destination
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $entry.source -Destination $target
}
foreach ($entry in $lock.payload_trees) {
    foreach ($file in Get-ChildItem -LiteralPath $entry.source -File -Recurse -Force) {
        $relative = $file.FullName.Substring($entry.source.Length).TrimStart('\')
        if ($relative -match '(^|[\\/])__pycache__([\\/]|$)' -or
            $file.Extension -in @('.pyc', '.pyo')) { continue }
        $target = Get-PayloadPath ($entry.destination + '/' + $relative)
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}
New-Item -ItemType Directory -Path (Join-Path $stage 'provenance') | Out-Null
Copy-Item -LiteralPath $lockPath -Destination (Join-Path $stage 'provenance\LOCAL-INPUTS.lock.json')
Copy-Item -LiteralPath (Join-Path $reference 'verify-inputs.py') -Destination (Join-Path $stage 'provenance\verify-inputs.py')
Copy-Item -LiteralPath (Join-Path $repo 'docs\qwenmonster\FIXED-WINDOWS-PACKAGE-20261024.md') -Destination (Join-Path $stage 'provenance\RUNBOOK.md')
rtk proxy $python -B (Join-Path $reference 'verify-inputs.py') --lock $lockPath --payload-root $stage
if ($LASTEXITCODE -ne 0) { throw 'Staged payload differs from fixed inputs' }
```

Expected layout:

```text
ovms/
  bin/ovms.exe + selected SDK/TBB/git2/curl/OpenCV DLLs
  python/                  # exact helper runtime, cache files excluded
  licenses/                # captured SDK, OVMS, OpenCV and curl notices
  LICENSE
  provenance/LOCAL-INPUTS.lock.json, verify-inputs.py, RUNBOOK.md
  setupvars.ps1            # add from the next block
```

Create `ovms\setupvars.ps1` with this package-relative content (not the original host-specific launchers):

```powershell
$env:PYTHONHOME = Join-Path $PSScriptRoot 'python'
$env:PYTHONDONTWRITEBYTECODE = '1'
Remove-Item Env:PYTHONPATH, Env:OPENVINO_LIB_PATHS -ErrorAction SilentlyContinue
$env:PATH = @(
    (Join-Path $PSScriptRoot 'bin'),
    $env:PYTHONHOME,
    "$env:SystemRoot\System32",
    $env:SystemRoot
) -join ';'
# No global SDK, C6, C:\opt, build-tree or Git DLL directory is appended.
```

Models and canonical chat templates remain external. Do not bundle private agent state, request history or a live model cache. Record model/template hashes in a separate acceptance receipt when model execution is authorized.

## Validate, archive, extract, and verify

First run only non-model smoke checks from the staged package, in a fresh PowerShell process. No server/model is started by `--version` or `--help`:

```powershell
. "$stage\setupvars.ps1"
& "$stage\bin\ovms.exe" --version
if ($LASTEXITCODE -ne 0) { throw 'Packaged version smoke failed' }
& "$stage\bin\ovms.exe" --help
if ($LASTEXITCODE -ne 0) { throw 'Packaged help smoke failed' }
& "$stage\python\python.exe" -B "$stage\provenance\verify-inputs.py" --payload-root $stage
if ($LASTEXITCODE -ne 0) { throw 'Package verification failed' }
```

These checks establish neither model inference nor dynamic plugin provenance. Save output and exit codes. For a fresh static audit, run the pinned `dumpbin.exe /DEPENDENTS` on every `bin\*.dll` and `bin\ovms.exe`; resolve imports against the payload/System32/API sets without adding development paths. Do not treat the saved import report as a fresh audit.

After smoke checks pass, create a package manifest covering **every** file, including setup/provenance scripts. This is the final payload manifest; it is separate from the source-input lock:

```powershell
function Get-Sha256([string]$path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($path)
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
$manifest = @(Get-ChildItem -LiteralPath $stage -File -Recurse -Force | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($stage.Length).TrimStart('\').Replace('\', '/')
        bytes = $_.Length
        sha256 = Get-Sha256 $_.FullName
    }
})
# Keep this manifest outside the payload while calculating it, then copy it in.
$stageParent = Split-Path -Parent $stage
$manifestPath = Join-Path $stageParent 'PACKAGE-FILES.json'
if (Test-Path -LiteralPath $manifestPath) { throw 'Manifest already exists' }
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $stage 'PACKAGE-FILES.json')
$zip = Join-Path $stageParent 'gemmamonster-2026.4-rc3b-parsers-eb76a74f-win-x64-fixed.zip'
if (Test-Path -LiteralPath $zip) { throw 'Archive already exists' }
& "$env:SystemRoot\System32\tar.exe" -a -c -f $zip -C $stageParent ovms
if ($LASTEXITCODE -ne 0) { throw 'Archive creation failed' }
$zipHash = Get-Sha256 $zip
"$zipHash  $([IO.Path]::GetFileName($zip))" | Set-Content -LiteralPath "$zip.sha256" -Encoding ASCII
```

The manifest excludes itself to avoid a self-hash cycle; retain the identical external manifest and ZIP SHA256. Extract the ZIP into another **new** directory. Validate every `PACKAGE-FILES.json` entry's relative path, size and SHA256, reject unexpected files except the manifest itself, then run the input verifier with `--payload-root` pointing at the extracted `ovms` directory. Repeat package-only smoke checks there. Extraction/repackaging must not overwrite the original EXE, SDK or model directories.

No Intel signing credentials are available as part of this recipe. Do not describe the fork EXE as Intel-signed; verify and record its actual signing state separately. Publication of the ZIP and sidecar is a later action after inspecting its concrete contents and receipts.

## Model smoke profile, only when separately authorized

After sourcing package `setupvars.ps1`, replace only the executable/environment parts of the original launchers with `$stage\bin\ovms.exe` and the package-relative Python/DLL paths. Keep these CLI parameters:

| Setting | Ornith | Gemma26 reference |
| --- | --- | --- |
| `--rest_bind_address` | `127.0.0.1` | `127.0.0.1` |
| `--rest_port` | `8137` | `18082` |
| `--model_name` | `ornith` | `gemma26` |
| `--model_path` | `C:\llm\models\ornith-35b-ov` | `C:\llm\models\OpenVINO\Wondernutts\gemma-4-26B-A4B-it-qat-q4_0-unquantized-uncensored-heretic-int4-ov-new` |
| `--tool_parser` / `--reasoning_parser` | `qwen3coder` / `qwen3` | `gemma4` / `gemma4` |
| Shared | `--task text_generation --pipeline_type VLM --target_device GPU --enable_tool_guided_generation true --max_num_seqs 1 --max_num_batched_tokens 4096 --cache_size 2` | Same resource settings |

This table documents the user's working launch profiles; it does not authorize running both models, rerunning Gemma acceptance, or changing the frozen Gemma parser. Serialize GPU model work. For package acceptance, save the process EXE path/hash and loaded module paths/hashes while the tested process is alive; require relevant non-system modules to come from the extracted package. Save raw unary/SSE responses before reserializing argument JSON, and exercise actual nonempty nested values. Keep the original live evidence and the new package receipt distinct.

## Rebuilding is a separate recovery route

If the known EXE is lost or fails its hash check, stop this fixed-byte packaging flow. Use an isolated checkout at the pinned source revision, the pinned RC1 headers/import libraries, Bazel/MSVC profile and original `build-ovms-test.ps1` / `build-ovms-live.ps1` as a recipe reference. Audit Bazel's `C:\opt` paths first. Do not reset the active checkout, use C6/latest dependencies, or label a new EXE with the old SHA256. Bit-for-bit rebuild reproducibility is not established. A rebuilt candidate needs its own hashes, parser tests, package checks and any required runtime acceptance.

Official packaging reference: [OVMS 2026.4.0 Windows packaging script](https://github.com/openvinotoolkit/model_server/blob/v2026.4.0/windows_create_package.bat) and [release pipeline](https://github.com/openvinotoolkit/model_server/blob/v2026.4.0/ci/build_test_release.groovy). This runbook adapts their runtime-plus-licenses ZIP layout to our explicit input lock; it does not invoke their mutable default dependency paths.
