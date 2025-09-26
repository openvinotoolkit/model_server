@echo off
set "BDBA_KEY=%1"
set "OVMS_PATH=..\%2"
cd repo_ci_infra

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

echo "BDBA_KEY=%BDBA_KEY%"
echo "OVMS_PATH=%OVMS_PATH%"

python binary_scans\ovms_bdba.py --key %BDBA_KEY% --type windows --build_dir %OVMS_PATH% --artifacts %zipname% --report_name %filename% 2>&1 | tee ..\win_bdba.log
for /f "tokens=2 delims=: " %%a in ('tail -n 3 win_bdba.log ^| findstr /c:"code":') do (
    if not "%%a"=="200" exit /b 1
)
deactivate

tar -cvf %OVMS_PATH%\ovms_windows_bdba_reports.zip -C ovms_windows_*
rm -rf %OVMS_PATH%\\%zipname%
