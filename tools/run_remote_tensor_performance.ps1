param(
    [string]$GTestFilter = "*",
    [switch]$BuildOnly,
    [switch]$NoTestCache,
    [int]$GTestRepeat = 1,
    [int]$BenchmarkTrialCount = 3
)

$ErrorActionPreference = "Stop"

if ($GTestRepeat -lt 1) {
    throw "GTestRepeat must be positive"
}
if ($BenchmarkTrialCount -lt 1) {
    throw "BenchmarkTrialCount must be positive"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$bazel = "C:\opt\bazel.exe"
$env:BAZEL_SH = "C:\opt\msys64\usr\bin\bash.exe"
$env:PATH = "C:\opt\python312;C:\opt\msys64\usr\bin;C:\opt\openvino\runtime\bin\intel64\Release;C:\opt\openvino\runtime\3rdparty\tbb\bin;C:\Program Files\gstreamer\1.0\msvc_x86_64\bin;$env:PATH"

if (-not (Test-Path $bazel)) {
    throw "Bazel was not found at $bazel"
}

function Invoke-Bazel {
    param([string[]]$Arguments)

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $bazel @Arguments
    $bazelExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($bazelExitCode -ne 0) {
        throw "Bazel exited with code $bazelExitCode"
    }
}

Push-Location $repoRoot
try {
    $startupArgs = @(
        "--output_user_root=C:\opt"
    )
    $commandArgs = @(
        "--config=win_mp_on_py_on",
        "--action_env", "OpenVINO_DIR=C:/opt/openvino/runtime/cmake",
        "--jobs=10"
    )
    $target = "//src/test/mediapipe/calculators:remote_tensor_performance_test"
    Invoke-Bazel ($startupArgs + @("build") + $commandArgs + @($target))
    if ($BuildOnly) {
        return
    }
    $testPathArgument = "--test_env=PATH=$env:PATH"
    $benchmarkTrialArgument = "--test_env=OVMS_PERF_TRIAL_COUNT=$BenchmarkTrialCount"
    $testCacheArgument = if ($NoTestCache) { "--nocache_test_results" } else { "" }
    $testArguments = @($testPathArgument, $benchmarkTrialArgument, "--test_arg=--gtest_filter=$GTestFilter",
        "--test_arg=--gtest_repeat=$GTestRepeat", "--test_output=streamed", "--test_summary=detailed", $target)
    if ($testCacheArgument) {
        $testArguments += $testCacheArgument
    }
    Invoke-Bazel ($startupArgs + @("test") + $commandArgs +
        $testArguments)
} finally {
    Pop-Location
}