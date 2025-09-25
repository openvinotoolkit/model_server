@echo off
set "OVMS_USER=%1"
set "OVMS_PASS=%2"
set "OVMS_FILES=%3"
REM
set PATH=%PATH%;C:\Jenkins\workspace\ovmsc\signfile
REM
python repo_signing\windows_signing\check_signing.py --user %OVMS_USER% --path %OVMS_FILES% --auto --verbose --print_all 2>&1 | tee win_sign.log