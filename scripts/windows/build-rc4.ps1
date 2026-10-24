param(
    [ValidateSet('off','on')][string]$Python = 'off',
    [string]$Output = 'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4',
    [int]$Jobs = 6
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$py = 'C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools'
$sdk = 'C:/gm/qwenmonster/semantic-refit-2026.4-sdk/extracted/openvino_genai_windows_2026.4.0.0rc1_x86_64/runtime/cmake'
$bazel = 'C:\gm\qwenmonster\semantic-refit-2026.4-tools\bazel.exe'
$config = "win_mp_on_py_$Python"
$dest = Join-Path $Output "build-py_$Python"
if (Test-Path "$dest\COMPLETE.json") { throw "Immutable build already exists: $dest" }
New-Item -ItemType Directory -Force $dest | Out-Null
$env:PATH = "$py;C:\Program Files\Git\bin;C:\Program Files\Git\usr\bin;" + $env:PATH
$env:BAZEL_SH = 'C:\Program Files\Git\bin\bash.exe'
$env:BAZEL_VS = 'C:\BuildTools'
$env:BAZEL_VC = 'C:\BuildTools\VC'
$env:BAZEL_VC_FULL_VERSION = '14.44.35207'
$env:PRODUCT_VERSION = '2026.4.0-rc4'
Set-Location $repo
$versionFile = Join-Path $repo 'src\version.hpp'
$original = [IO.File]::ReadAllBytes($versionFile)
try {
    & "$py\python.exe" windows_set_ovms_version.py "--config=$config" $sdk
    if ($LASTEXITCODE) { throw 'Version stamping failed' }
    $flags = @("--config=$config", "--action_env=OpenVINO_DIR=$sdk", "--action_env=PYTHON_BIN_PATH=$py/python.exe", "--action_env=PYTHON_LIB_PATH=$py/Lib/site-packages", "--repo_env=PYTHON_BIN_PATH=$py/python.exe", "--repo_env=PYTHON_LIB_PATH=$py/Lib/site-packages", "--python_path=$py/python.exe", "--jobs=$Jobs", '--verbose_failures')
    & $bazel --output_user_root=C:\r20264 build @flags //src:ovms //src:ovms_test
    if ($LASTEXITCODE) { throw "Bazel failed: $LASTEXITCODE" }
    New-Item -ItemType Directory -Force "$dest\bin" | Out-Null
    Copy-Item "$repo\bazel-bin\src\ovms.exe", "$repo\bazel-bin\src\ovms_test.exe", "$repo\bazel-bin\src\*.dll" "$dest\bin"
    if ($Python -eq 'on') {
        Copy-Item "$repo\bazel-bin\src\python\binding\pyovms.pyd" $dest
    }
    Copy-Item $versionFile "$dest\version.hpp"
    $files = Get-ChildItem $dest -File -Recurse | ForEach-Object {
        $hash = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($_.FullName)
        try { $sha = [BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-','').ToLower() } finally { $stream.Dispose(); $hash.Dispose() }
        [ordered]@{path=$_.FullName.Substring($dest.Length+1); bytes=$_.Length; sha256=$sha}
    }
    [ordered]@{source_commit=(& git rev-parse HEAD); config=$config; flags=$flags; files=@($files); status='BUILD_PASS'} | ConvertTo-Json -Depth 8 | Set-Content "$dest\COMPLETE.json" -Encoding UTF8
} finally {
    [IO.File]::WriteAllBytes($versionFile, $original)
}
