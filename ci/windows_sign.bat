@echo off
set "OVMS_USER=%1"
set "OVMS_FILES=..\..\%2"
set "PYTHON=%3"
set PATH=%PATH%;C:\Jenkins\workspace\ovmsc\signfile;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64

cd sdl_repo\windows_signing

if /I "%PYTHON%"=="1" (
    set "PYTHON_OPT=--python"
) else (
    set "PYTHON_OPT="
)

python check_signing.py --user=%OVMS_USER% --path=%OVMS_FILES% %PYTHON_OPT% --auto --verbose --print_all 2>&1 | tee ..\..\win_sign.log
python check_signing.py --zip --path=%OVMS_FILES% %PYTHON_OPT% --auto

for %%f in (ovms_windows_python_*) do (
    copy "%%f" "%OVMS_FILES%"
)
for /f "tokens=* delims=" %%a in ('type ..\..\win_sign.log ^| tail -n 1') do (
    echo %%a | findstr /C:"[ OK ]" >nul
    if not errorlevel 1 (
        exit /b 0
    ) else (
        exit /b 1
    )
)