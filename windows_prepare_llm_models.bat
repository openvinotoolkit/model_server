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
setlocal

if "%~1"=="" (
  echo Error: No directory specified.
  exit /b 1
)

:: Create a link to preexported models on CI workers
IF /I EXIST c:\opt\llm_testing__ (
    rmdir /S /Q "%~1"
    mklink /d "%~1" c:\opt\llm_testing
    echo Created link to existing in c:\opt\llm_testing. Skipping downloading models.
    exit /b 0
)

set "EMBEDDING_MODEL=thenlper/gte-small"
set "RERANK_MODEL=BAAI/bge-reranker-base"
set "TEXT_GENERATION_MODEL=facebook/opt-125m"

if exist "%~1\%TEXT_GENERATION_MODEL%" if exist "%~1\%EMBEDDING_MODEL%" if exist "%~1\%RERANK_MODEL%" (
  echo Models directory %~1 exists. Skipping downloading models.
  exit /b 0
)

echo Downloading LLM testing models to directory %~1
set "PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
set PYTHONHOME=
"C:\Program Files\Python310\python.exe" -m venv .venv
.\.venv\Scripts\activate
set
python -m pip install --upgrade pip
pip install -U -r demos\common\export_models\requirements_win.txt

if not exist "%~1" mkdir "%~1"

if exist "%~1\%TEXT_GENERATION_MODEL%" (
  echo Models directory %~1\%TEXT_GENERATION_MODEL% exists. Skipping downloading models.
) else (
  python demos\common\export_models\export_model.py text_generation --source_model "%TEXT_GENERATION_MODEL%" --weight-format int8 --model_repository_path %~1
)

if exist "%~1\%EMBEDDING_MODEL%" (
  echo Models directory %~1\%EMBEDDING_MODEL% exists. Skipping downloading models.
) else (
  python demos\common\export_models\export_model.py embeddings --source_model "%EMBEDDING_MODEL%" --weight-format int8 --model_repository_path %~1
)

if exist "%~1\%RERANK_MODEL%" (
  echo Models directory %~1\%RERANK_MODEL% exists. Skipping downloading models.
) else (
  python demos\common\export_models\export_model.py rerank --source_model "%RERANK_MODEL%" --weight-format int8 --model_repository_path %~1
)

endlocal
