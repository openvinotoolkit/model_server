@echo off
set "BDBA_KEY=%1"
set "OVMS_PATH=..\%2"
set "CONFIG_PATH=..\%3"
cd repo_ci_infra

del ovms_windows*

python -m venv venv

call venv\Scripts\activate

python -m pip install --upgrade pip

if exist requirements.txt (
    pip install -r requirements.txt
)

for /f "tokens=2 delims==." %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set datestamp=%datetime:~0,8%
set timestamp=%datetime:~8,4%
set filename=ovms_windows_%datestamp%_%timestamp%
set zipname="%filename%.zip"

copy %OVMS_PATH%\\ovms.zip %OVMS_PATH%\\%zipname%
if errorlevel 1 (
    echo Failed to copy %OVMS_PATH%\ovms.zip to %OVMS_PATH%\%zipname%.
    exit /b 1
)

echo "BDBA_KEY=%BDBA_KEY%"
echo "OVMS_PATH=%OVMS_PATH%"

python binary_scans\ovms_bdba.py --key %BDBA_KEY% --config_dir=%CONFIG_PATH% --type windows --build_dir %OVMS_PATH% --artifacts %zipname% --report_name %filename% 2>&1 | tee ..\win_bdba.log
if errorlevel 1 exit /b %errorlevel%

tar -a -c -f ..\ovms_windows_bdba_reports.zip ovms_windows*

del "%OVMS_PATH%\%zipname%"
