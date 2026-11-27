# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

param(
    [string]$LLMDir = "",
    [switch]$PrepareLLM
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Install-ONNX {
    $onnxDir = Join-Path $repoRoot "demos\using_onnx_model\python\workspace\resnet50-onnx"
    $modelDir = Join-Path $onnxDir "1"
    $onnxModel = Join-Path $modelDir "resnet50-caffe2-v1-9.onnx"

    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    if (Test-Path -LiteralPath $onnxModel) {
        Write-Host "Reusing ONNX model: $onnxModel"
        return
    }

    Write-Host "Downloading ONNX model: $onnxModel"
    $url = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx"
    Invoke-WebRequest -Uri $url -OutFile $onnxModel
}

function Install-Paddle {
    $paddleDir = Join-Path $repoRoot "demos\classification_using_paddlepaddle_model\python"
    $modelDir = Join-Path $paddleDir "model\1"
    $pdmodel = Join-Path $modelDir "inference.pdmodel"
    $pdiparams = Join-Path $modelDir "inference.pdiparams"

    if ((Test-Path -LiteralPath $pdmodel) -and (Test-Path -LiteralPath $pdiparams)) {
        Write-Host "Reusing PaddlePaddle model: $modelDir"
        return
    }

    Write-Host "Downloading PaddlePaddle model: $modelDir"
    $downloadScript = Join-Path $paddleDir "download_model.py"
    Push-Location $paddleDir
    try {
        & python $downloadScript
        if ($LASTEXITCODE -ne 0) {
            throw "PaddlePaddle model download failed"
        }
    } finally {
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $pdmodel) -or -not (Test-Path -LiteralPath $pdiparams)) {
        throw "PaddlePaddle model files are missing: $modelDir"
    }
}

function Install-TensorFlow {
    $modelDir = Join-Path $repoRoot "src\test\dummy_tflite\1"
    if (-not (Test-Path -LiteralPath (Join-Path $modelDir "dummy.tflite"))) {
        throw "Missing TensorFlow Lite fixture: $modelDir\dummy.tflite"
    }
    Write-Host "Reusing TensorFlow Lite model: $modelDir"
}

Install-ONNX
Install-Paddle
Install-TensorFlow

if ($PrepareLLM) {
    if ([string]::IsNullOrWhiteSpace($LLMDir)) {
        $LLMDir = Join-Path $repoRoot "src\test\llm_testing"
    }

    $prepareLlmScript = Join-Path $repoRoot "windows_prepare_llm_models.bat"
    if (-not (Test-Path -LiteralPath $prepareLlmScript)) {
        throw "Missing Windows LLM model preparation script: $prepareLlmScript"
    }

    Write-Host "Preparing LLM fixtures: $LLMDir"
    & cmd.exe /d /c "`"$prepareLlmScript`" `"$LLMDir`""
    if ($LASTEXITCODE -ne 0) {
        throw "windows_prepare_llm_models.bat failed"
    }
}
