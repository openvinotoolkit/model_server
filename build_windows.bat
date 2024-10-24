@echo on
set vsDevPath="C:\Program\ Files\ (x86)\Microsoft\ Visual\ Studio\2019\Professional\Common7\Tools\vsdevcmd"
set setPath=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\bin\Roslyn;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\HTML Help Workshop;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\devinit;C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x86;C:\Program Files (x86)\Windows Kits\10\bin\x86;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;S:\Python27;S:\Python27\Scripts;s:\telnet;S:\Tcl\bin;C:\Program Files\SafeNet\LunaClient\win32;C:\Program Files\SafeNet\LunaClient\;C:\ProgramData\chocolatey\bin;C:\Program Files\clinfo\;C:\Program Files\NVIDIA Corporation\Nsight Compute 2022.3.0\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\NVIDIA cuTENSOR\lib\11.0;C:\Program Files\NVIDIA zlib123dllx64\dll_x64;C:\Program Files\NVIDIA cuDNN\bin;C:\Utils\;C:\Program Files\Git\cmd;C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\usr\bin;C:\Ninja;C:\Program Files\CMake\bin;C:\Program Files\7-zip;C:\moviTools\21.07.5\win32\bin;C:\Program Files\postgresql\bin;C:\opt\Python39\Scripts\;C:\opt\Python39\;C:\opencl\install\;C:\opencl\;C:\Users\rasapala\AppData\Local\Microsoft\WindowsApps;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;c:\opt
set buildCommand=bazel build --config=windows --jobs=64 --verbose_failures --define CLOUD_DISABLE=1 --define MEDIAPIPE_DISABLE=1 --define PYTHON_DISABLE=1 //src:ovms ^> compilation.log 2^>^&1
set envPath=environment.log

:: Start VS 2019 Developer command line
%vsDevPath%

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set PATH=%setPath%

:: Log all environment variables
set ^> %envPath%

:: Start bazel build
%buildCommand%