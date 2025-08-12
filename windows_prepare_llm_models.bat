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
set "TEXT_GENERATION_MODEL=facebook/opt-125m"
set "VLM_MODEL=OpenGVLab/InternVL2-1B"

:: Models for tools testing. Only tokenizers are downloaded.
set "QWEN3_MODEL=Qwen/Qwen3-8B"
set "LLAMA3_MODEL=meta-llama/Llama-3.1-8B-Instruct"
set "HERMES3_MODEL=NousResearch/Hermes-3-Llama-3.1-8B"
set "PHI4_MODEL=microsoft/Phi-4-mini-instruct"

set MODELS_LIST=%TEXT_GENERATION_MODEL% %EMBEDDING_MODEL% %EMBEDDING_MODEL%\ov %RERANK_MODEL% %VLM_MODEL% %QWEN3_MODEL% %LLAMA3_MODEL% %HERMES3_MODEL% %PHI4_MODEL%

set "ALL_EXIST=1"
for %%M in ("%MODELS_LIST%") do (
  if not exist "%~1\%%~M" (
    set "ALL_EXIST=0"
  )
)

if "!ALL_EXIST!"=="1" (
  echo All required models exist in %~1. Skipping downloading models.
  exit /b 0
)

echo Downloading LLM testing models to directory %~1
set "PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
set "PYTHONPATH="
C:\opt\Python312\python.exe -m venv .venv
if !errorlevel! neq 0 exit /b !errorlevel!
call .\.venv\Scripts\Activate.bat
if !errorlevel! neq 0 exit /b !errorlevel!
set
python -m pip install --upgrade pip
if !errorlevel! neq 0 exit /b !errorlevel!
BUILD_CUDA_EXT=0 pip install -U -r demos\common\export_models\requirements.txt
if !errorlevel! neq 0 exit /b !errorlevel!

if not exist "%~1" mkdir "%~1"

if exist "%~1\%TEXT_GENERATION_MODEL%" (
  echo Models directory %~1\%TEXT_GENERATION_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading text generation model to %~1\%TEXT_GENERATION_MODEL% directory.
  python demos\common\export_models\export_model.py text_generation --source_model "%TEXT_GENERATION_MODEL%" --weight-format int8 --model_repository_path %~1
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%EMBEDDING_MODEL%" (
  echo Models directory %~1\%EMBEDDING_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading embeddings model to %~1\%EMBEDDING_MODEL% directory.
  python demos\common\export_models\export_model.py embeddings --source_model "%EMBEDDING_MODEL%" --weight-format int8 --model_repository_path %~1
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%EMBEDDING_MODEL%\ov" (
  echo Models directory %~1\%EMBEDDING_MODEL%\ov exists. Skipping downloading models.
) else (
  echo Downloading embeddings model to %~1\%EMBEDDING_MODEL%\ov directory.
  python demos\common\export_models\export_model.py embeddings_ov --source_model "%EMBEDDING_MODEL%" --weight-format int8 --model_repository_path %~1 --model_name "%EMBEDDING_MODEL%\ov"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%RERANK_MODEL%" (
  echo Models directory %~1\%RERANK_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading rerank model to %~1\%RERANK_MODEL% directory.
  python demos\common\export_models\export_model.py rerank --source_model "%RERANK_MODEL%" --weight-format int8 --model_repository_path %~1
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%RERANK_MODEL%\ov" (
  echo Models directory %~1\%RERANK_MODEL%\ov exists. Skipping downloading models.
) else (
  echo Downloading rerank model to %~1\%RERANK_MODEL%\ov directory.
  python demos\common\export_models\export_model.py rerank_ov --source_model "%RERANK_MODEL%" --weight-format int8 --model_repository_path %~1 --model_name "%RERANK_MODEL%\ov"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%VLM_MODEL%" (
  echo Models directory %~1\%VLM_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading visual language model to %~1\%VLM_MODEL% directory.
  python demos\common\export_models\export_model.py text_generation --source_model "%VLM_MODEL%" --weight-format int4 --kv_cache_precision u8 --model_repository_path %~1
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%QWEN3_MODEL%" (
  echo Models directory %~1\%QWEN3_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading tokenizer and detokenizer for Qwen3 model to %~1\%QWEN3_MODEL% directory.
  mkdir "%~1\%QWEN3_MODEL%"
  convert_tokenizer "%QWEN3_MODEL%" --with_detokenizer -o "%~1\%QWEN3_MODEL%"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%LLAMA3_MODEL%" (
  echo Models directory %~1\%LLAMA3_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading tokenizer and detokenizer for Llama3.1 model to %~1\%LLAMA3_MODEL% directory.
  mkdir "%~1\%LLAMA3_MODEL%"
  convert_tokenizer "%LLAMA3_MODEL%" --with_detokenizer -o "%~1\%LLAMA3_MODEL%"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%HERMES3_MODEL%" (
  echo Models directory %~1\%HERMES3_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading tokenizer and detokenizer for Hermes3 model to %~1\%HERMES3_MODEL% directory.
  mkdir "%~1\%HERMES3_MODEL%"
  convert_tokenizer "%HERMES3_MODEL%" --with_detokenizer -o "%~1\%HERMES3_MODEL%"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

if exist "%~1\%PHI4_MODEL%" (
  echo Models directory %~1\%PHI4_MODEL% exists. Skipping downloading models.
) else (
  echo Downloading tokenizer and detokenizer for Phi-4 model to %~1\%PHI4_MODEL% directory.
  mkdir "%~1\%PHI4_MODEL%"
  convert_tokenizer "%PHI4_MODEL%" --with_detokenizer -o "%~1\%PHI4_MODEL%"
  if !errorlevel! neq 0 exit /b !errorlevel!
)

endlocal
