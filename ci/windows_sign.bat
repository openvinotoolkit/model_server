@echo off
set "OVMS_USER=%1"
set "OVMS_PASS=%2"
set "OVMS_FILES=%3"
set "PYTHON=%4"
set PATH=%PATH%;C:\Jenkins\workspace\ovmsc\signfile

if /I "%PYTHON%"=="1" (
    set "PYTHON_OPT=--python"
) else (
    set "PYTHON_OPT="
)

python repo_signing\windows_signing\check_signing.py --user=%OVMS_USER% --path=%OVMS_FILES% %PYTHON_OPT% --auto --verbose --print_all 2>&1 | tee win_sign.log