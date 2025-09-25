@echo off
set "OVMS_USER=%1"
set "OVMS_PASS=%2"
set "OVMS_FILES=%3"
REM
curl https://github.com/intel-innersource/applications.security.edss.docs.signfile-releases/releases/download/v4.0.180/signfile-win-x64-4.0.180.zip -o signfile-win-x64-4.0.180.zip
REM
tar -xf signfile-win-x64-4.0.180.zip
REM
set PATH=%PATH%;%CD%\signfile-win-x64-4.0.180
REM
python repo_signing\windows_signing\check_signing.py --user %OVMS_USER% --path %OVMS_FILES% --auto --verbose --print_all 2>&1 | tee win_sign.log