@echo off
set "BDBA_KEY=%1"
set "OVMS_PATH=%2"

python -m venv venv

call venv\Scripts\activate

python -m pip install --upgrade pip

if exist repo_ci_infra\requirements.txt (
    pip install -r repo_ci_infra\requirements.txt
)

for /f "tokens=2 delims==." %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set datestamp=%datetime:~0,8%
set timestamp=%datetime:~8,4%
set filename=ovms_%datestamp%_%timestamp%.zip

copy %OVMS_PATH%\\ovms.zip %OVMS_PATH%\\%filename%

echo "BDBA_KEY=%BDBA_KEY%"
echo "OVMS_PATH=%OVMS_PATH%"

python repo_ci_infra\binary_scans\ovms_bdba.py --key %BDBA_KEY% --type windows --build_dir %OVMS_PATH% --artifacts %filename% --report_name %filename% 2>&1 | tee win_bdba.log

deactivate

tar -cvf %OVMS_PATH%\ovms_windows_bdba_reports.zip -C %OVMS_PATH% ovms_windows_*