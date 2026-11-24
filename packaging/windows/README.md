# OVMS Windows Packaging

This directory contains the Windows installer and runtime-control scripts for OpenVINO Model Server.

## Contents

- `installer/OVMSInstaller.iss` — Inno Setup installer definition.
- `installer/build_installer.ps1` — Builds `dist/windows/OVMS-Setup.exe` from the packaged OVMS directory.
- `scripts/` — Runtime configuration, PATH, service, repair, start, stop, and uninstall helpers.

## Build

Build the OVMS package first, then build the installer:

```text
.\windows_install_build_dependencies.bat
.\windows_build.bat
.\windows_create_package.bat opt --with_python
.\packaging\windows\installer\build_installer.ps1
```

Expected installer output:

```text
dist/windows/OVMS-Setup.exe
```

## Python modes

The installer supports three Python modes:

- System Python 3.14
- Embedded/bundled Python
- No Python

Python configuration is performed by `scripts/configure-ovms.ps1`.
