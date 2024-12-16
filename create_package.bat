md dist\windows
copy bazel-bin\src\ovms.exe dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy  %cd%/bazel-out/x64_windows-opt/bin/src/python39.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy c:\opt\openvino\runtime\bin\intel64\Release\*.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy c:\opt\openvino\runtime\3rdparty\tbb\bin\tbb12.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy  %cd%\bazel-out\x64_windows-opt\bin\src\opencv_world4100.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
tar -czf dist\ovms.zip dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%