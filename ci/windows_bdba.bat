@echo off
set "BDBA_KEY=%1"
REM 
git clone https://github.com/intel-innersource/frameworks.ai.openvino.ci.infrastructure repo_ci_infra

REM
cd repo_ci_infra

REM
python -m venv venv

REM
call venv\Scripts\activate

REM
python -m pip install --upgrade pip

REM
if exist requirements.txt (
    pip install -r requirements.txt
)

REM
for /f "tokens=2 delims==." %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set datestamp=%datetime:~0,8%
set timestamp=%datetime:~8,4%
set filename=ovms_%datestamp%_%timestamp%.zip

REM
copy dist\windows\ovms.zip dist\windows\%filename%

REM
python binary_scans\ovms_bdba.py --key %BDBA_KEY% --type windows --build_dir dist\windows --artifacts dist\windows\%filename% --report_name %filename% > bdba_windows.log 2>&1