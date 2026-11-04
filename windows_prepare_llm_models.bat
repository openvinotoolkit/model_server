::
:: Copyright 2024 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under

@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="" (
  echo Error: No directory specified.
  exit /b 1
)

:: Create a link to preexported models on CI workers
IF /I EXIST c:\opt\llm_testing (
    rmdir /S /Q "%~1"
    mklink /d "%~1" c:\opt\llm_testing
    echo Created link to existing in c:\opt\llm_testing. Skipping downloading models.
)

set "EMBEDDING_MODEL=OpenVINO/bge-base-en-v1.5-int8-ov"
set "RERANK_MODEL=OpenVINO/bge-reranker-base-int8-ov"
:: Dummy cyclic models used by LLM/VLM tests instead of real downloaded models;
set "DUMMY_LLM_MODEL=mzeglars/dummy-cyclic-gpt2-ov"
set "DUMMY_VLM_MODEL=mzeglars/dummy-cyclic-llava-ov"
set "TTS_MODEL=OpenVINO/Kokoro-82M-int8-ov"
set "TEXT_GENERATION_MODEL=OpenVINO/LFM2.5-350M-int8-ov"
set "STT_MODEL=OpenVINO/whisper-tiny-int4-ov"

:: Models for tools testing. Only tokenizers are downloaded.
set "QWEN3_MODEL=OpenVINO/Qwen3-8B-int4-ov"
set "LLAMA3_MODEL=OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"
set "HERMES3_MODEL=NousResearch/Hermes-3-Llama-3.1-8B"
set "PHI4_MODEL=OpenVINO/Phi-4-mini-instruct-int4-ov"
set "MISTRAL_MODEL=OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov"
set "GPTOSS_MODEL=OpenVINO/gpt-oss-20b-int4-ov"
set "DEVSTRAL_MODEL=unsloth/Devstral-Small-2507"
set "LFM2_MODEL=OpenVINO/LFM2-24B-A2B-int4-ov"
set "LFM25_MODEL=OpenVINO/LFM2.5-8B-A1B-int4-ov"
set "GEMMA4_MODEL=OpenVINO/gemma-4-E4B-it-int4-ov"
set "MINICPM5_MODEL=OpenVINO/MiniCPM5-1B-int4-ov"
set "ONYX_MODEL=OpenVINO/Muse-Glimmer-30B-int4-ov"

echo Downloading LLM testing models to directory %~1
set "PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
set "PYTHONPATH="
C:\opt\Python312\python.exe -m venv .venv
if !errorlevel! neq 0 exit /b !errorlevel!
call .\.venv\Scripts\Activate.bat
if !errorlevel! neq 0 exit /b !errorlevel!
python -m pip install --upgrade pip
if !errorlevel! neq 0 exit /b !errorlevel!
pip install -U nncf sentence_transformers==5.3.0 sentencepiece requests protobuf==7.35.0 kokoro openvino-tokenizers
if !errorlevel! neq 0 exit /b !errorlevel!

if not exist "%~1" mkdir "%~1"


:: Download exported models
call :download_hf_model_if_missing "%TEXT_GENERATION_MODEL%" "%~1" "openvino_model.xml"
if !errorlevel! neq 0 exit /b !errorlevel!
call :download_hf_model_if_missing "%TTS_MODEL%" "%~1" "openvino_model.xml"
if !errorlevel! neq 0 exit /b !errorlevel!
call :download_hf_model_if_missing "%STT_MODEL%" "%~1" "openvino_tokenizer.bin"
if !errorlevel! neq 0 exit /b !errorlevel!
call :download_hf_model_if_missing "%EMBEDDING_MODEL%" "%~1" "openvino_tokenizer.bin"
if !errorlevel! neq 0 exit /b !errorlevel!
call :download_hf_model_if_missing "%RERANK_MODEL%" "%~1" "openvino_tokenizer.bin"
if !errorlevel! neq 0 exit /b !errorlevel!
call :download_hf_model "%DUMMY_LLM_MODEL%" "%~1"
call :download_hf_model "%DUMMY_VLM_MODEL%" "%~1"

:: Download tokenizers for tools testing
call :download_openvino_tokenizer "%QWEN3_MODEL%" "%~1"
call :download_openvino_tokenizer "%LLAMA3_MODEL%" "%~1"
call :download_tokenizer "%HERMES3_MODEL%" "%~1\%HERMES3_MODEL%"
call :download_openvino_tokenizer "%PHI4_MODEL%" "%~1"
call :download_openvino_tokenizer "%MISTRAL_MODEL%" "%~1"
call :download_openvino_tokenizer "%GPTOSS_MODEL%" "%~1"
call :download_tokenizer "%DEVSTRAL_MODEL%" "%~1\%DEVSTRAL_MODEL%"
call :download_openvino_tokenizer "%LFM2_MODEL%" "%~1"
call :download_openvino_tokenizer "%LFM25_MODEL%" "%~1"
call :download_openvino_tokenizer "%MINICPM5_MODEL%" "%~1"
call :download_openvino_tokenizer "%ONYX_MODEL%" "%~1"
call :download_openvino_tokenizer "%GEMMA4_MODEL%" "%~1"

exit /b 0

:download_hf_model_if_missing
set "model=%~1"
set "repository=%~2"
set "required_file=%~3"

if not exist "%repository%\%model%\%required_file%" (
  echo Downloading %model% to %repository%\%model% directory.
  hf download "%model%" --local-dir "%repository%\%model%"
  if !errorlevel! neq 0 exit /b !errorlevel!
) else (
  echo Model file %repository%\%model%\%required_file% exists. Skipping downloading model.
)
if not exist "%repository%\%model%\%required_file%" (
  echo [ERROR] Model file %repository%\%model%\%required_file% does not exist.
  exit /b 1
)
exit /b 0

:download_hf_model
set "model=%~1"
set "repository=%~2"

REM Dummy models are still actively being iterated on - always re-download a fresh copy
REM instead of skipping when already present, so local/CI caches can't go stale.
echo Downloading %model% to %repository%\%model% directory.
hf download "%model%" --local-dir "%repository%\%model%"
if !errorlevel! neq 0 exit /b !errorlevel!
if not exist "%repository%\%model%\openvino_tokenizer.bin" (
  echo [ERROR] Model file %repository%\%model%\openvino_tokenizer.bin does not exist.
  exit /b 1
)
exit /b 0

:download_openvino_tokenizer
set "model=%~1"
set "repository=%~2"

if not exist "%repository%\%model%\openvino_tokenizer.bin" (
  echo Downloading tokenizer and detokenizer for %model% model to %repository%\%model% directory.
  mkdir "%repository%\%model%"
  hf download "%model%" --local-dir "%repository%\%model%" --include *tokenizer*
) else (
  echo Models file %repository%\%model%\openvino_tokenizer.bin exists. Skipping downloading models.
)

:: Helper subroutine to download tokenizers
:download_tokenizer
set "model=%~1"
set "check_path=%~2"

if exist "%check_path%" (
  echo Models file %check_path% exists. Skipping downloading models.
) else (
  echo Downloading tokenizer and detokenizer for %model% model to %check_path% directory.
  mkdir "%check_path%"
  convert_tokenizer "%model%" --with_detokenizer -o "%check_path%"
  if !errorlevel! neq 0 exit /b !errorlevel!
)
if not exist "%check_path%\openvino_tokenizer.bin" (
  echo Models file %check_path%\openvino_tokenizer.bin does not exist.
  exit /b 1
)
exit /b 0

endlocal
