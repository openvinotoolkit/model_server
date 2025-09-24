@echo off
REM
curl https://raw.githubusercontent.com/intel-innersource/frameworks.ai.openvino.model-server.bdba/refs/heads/main/windows_signing/check_signing.py?token=GHSAT0AAAAAADGTIRKXLRUCX2YYYKZY5PLI2GRGVLA -o check_signing.py
REM
curl https://github.com/intel-innersource/applications.security.edss.docs.signfile-releases/releases/download/v4.0.180/signfile-win-x64-4.0.180.zip -o signfile-win-x64-4.0.180.zip
REM
tar -xf signfile-win-x64-4.0.180.zip
REM
set PATH=%PATH%;%CD%\signfile-win-x64-4.0.180
REM
python check_signing.py --user %USERNAME% --path %OVMS_FILES% --auto --verbose --print_all 2>&1 | tee win_sign.log