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

set "EMBEDDING_MODEL=thenlper/gte-small"
set "RERANK_MODEL=BAAI/bge-reranker-base"
set "TEXT_GENERATION_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct"
set "FACEBOOK_MODEL=facebook/opt-125m"
set "VLM_MODEL=OpenGVLab/InternVL2-1B"
set "TTS_MODEL=microsoft/speecht5_tts"
set "STT_MODEL=openai/whisper-tiny"

:: Models for tools testing. Only tokenizers are downloaded.
set "QWEN3_MODEL=Qwen/Qwen3-8B"
set "LLAMA3_MODEL=unsloth/Llama-3.1-8B-Instruct"
set "HERMES3_MODEL=NousResearch/Hermes-3-Llama-3.1-8B"
set "PHI4_MODEL=microsoft/Phi-4-mini-instruct"
set "MISTRAL_MODEL=mistralai/Mistral-7B-Instruct-v0.3"
set "GPTOSS_MODEL=openai/gpt-oss-20b"
set "DEVSTRAL_MODEL=unsloth/Devstral-Small-2507"

echo Downloading LLM testing models to directory %~1
set "PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
set "PYTHONPATH="
C:\opt\Python312\python.exe -m venv .venv
if !errorlevel! neq 0 exit /b !errorlevel!
call .\.venv\Scripts\Activate.bat
if !errorlevel! neq 0 exit /b !errorlevel!
python -m pip install --upgrade pip
if !errorlevel! neq 0 exit /b !errorlevel!
pip install -U -r demos\common\export_models\requirements.txt
if !errorlevel! neq 0 exit /b !errorlevel!

if not exist "%~1" mkdir "%~1"


:: Export models
call :download_export_model "%TTS_MODEL%" "text2speech" "--weight-format int4" "--vocoder microsoft/speecht5_hifigan" "%~1"
call :download_export_model "%STT_MODEL%" "speech2text" "--weight-format int4" "%~1"
call :download_export_model "%VLM_MODEL%" "text_generation" "--weight-format int4" "%~1"
call :download_export_model "%TEXT_GENERATION_MODEL%" "text_generation" "--weight-format int8" "%~1"
call :download_export_model "%FACEBOOK_MODEL%" "text_generation" "--weight-format int8" "%~1"
call :download_export_model "%RERANK_MODEL%" "rerank_ov" "--weight-format int8 --model_name %RERANK_MODEL%\ov" "%~1"
call :download_export_model "%EMBEDDING_MODEL%" "embeddings_ov" "--weight-format int8 --model_name %EMBEDDING_MODEL%\ov" "%~1"

if not exist "%~1\%FACEBOOK_MODEL%\chat_template.jinja" (
    echo Copying dummy chat template to %FACEBOOK_MODEL% model directory.
    copy /Y "src\test\llm\dummy_facebook_template.jinja" "%~1\%FACEBOOK_MODEL%\chat_template.jinja"
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:: Download tokenizers for tools testing
call :download_tokenizer "%QWEN3_MODEL%" "%~1\%QWEN3_MODEL%"
call :download_tokenizer "%LLAMA3_MODEL%" "%~1\%LLAMA3_MODEL%"
call :download_tokenizer "%HERMES3_MODEL%" "%~1\%HERMES3_MODEL%"
call :download_tokenizer "%PHI4_MODEL%" "%~1\%PHI4_MODEL%"
call :download_tokenizer "%MISTRAL_MODEL%" "%~1\%MISTRAL_MODEL%"
call :download_tokenizer "%GPTOSS_MODEL%" "%~1\%GPTOSS_MODEL%"
call :download_tokenizer "%DEVSTRAL_MODEL%" "%~1\%DEVSTRAL_MODEL%"

exit /b 0

:: Helper subroutine to download export models
:download_export_model
set "model=%~1"
set "model_type=%~2"
set "export_args=%~3"
set "repository=%~4"

if not exist "%repository%\%model%\openvino_tokenizer.bin" (
  echo Downloading %model_type% model to %repository%\%model% directory.
  python demos\common\export_models\export_model.py %model_type% --source_model "%model%" %export_args% --model_repository_path %repository%
) else (
  echo Models file %repository%\%model%\openvino_tokenizer.bin exists. Skipping downloading models.
)
exit /b 0

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
