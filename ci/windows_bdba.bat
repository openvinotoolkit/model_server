@echo off
set "BDBA_KEY=%1"
set "OVMS_PATH=%2"
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
copy %OVMS_PATH%\\ovms.zip %OVMS_PATH%\\%filename%

REM
python binary_scans\ovms_bdba.py --key %BDBA_KEY% --type windows --build_dir %OVMS_PATH% --artifacts %OVMS_PATH%\%filename% --report_name %filename% 2>&1 | tee ..\win_bdba.log
REM
deactivate
REM
tar -cvf %2\ovms_windows_bdba_reports.zip -C %2 ovms_windows_*