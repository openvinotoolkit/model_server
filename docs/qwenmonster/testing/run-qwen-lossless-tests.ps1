param(
    [string]$Filter = '*VariousArgumentTypes*',
    [string]$Label = 'focused',
    [string]$ReportDirectory = 'C:\gm\qwenmonster\semantic-refit-2026.4-hub\tasks\C12\reports'
)
$ErrorActionPreference = 'Stop'
$py = 'C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools'
$env:PYTHONHOME = $py
$env:PATH = @(
    $py,
    'C:\gm\qwenmonster\semantic-refit-2026.4-sdk\extracted\openvino_genai_windows_2026.4.0.0rc1_x86_64\runtime\bin\intel64\Release',
    'C:\gm\qwenmonster\semantic-refit-2026.4-sdk\extracted\openvino_genai_windows_2026.4.0.0rc1_x86_64\runtime\3rdparty\tbb\bin',
    'C:\opt\opencv_4.14.0\bin',
    'C:\opt\curl-8.22.0_2-win64-mingw\bin',
    'C:\Program Files\Git\mingw64\bin',
    $env:PATH
) -join ';'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
Set-Location -LiteralPath $repoRoot
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
& .\bazel-bin\src\ovms_test.exe "--gtest_filter=$Filter" "--gtest_output=xml:$ReportDirectory\qwen-lossless-$Label.xml" *> "$ReportDirectory\qwen-lossless-$Label.log"
$testExit = $LASTEXITCODE
Get-Content "$ReportDirectory\qwen-lossless-$Label.log" -Tail 10
exit $testExit
