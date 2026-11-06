# OVMS Windows Packaging

This folder contains the Windows installer, uninstaller, server-control scripts,
and lightweight tray app for OpenVINO Model Server.

The first implementation intentionally keeps the full configuration UI out of
scope. It provides:

- An Inno Setup installer that consumes `dist/windows/ovms`.
- A standard Windows uninstaller.
- First-run settings and install markers under `C:\ProgramData\OVMS`.
- A minimal tray app with:
  - Open OVMS Manager
  - Start Server
  - Stop Server

The installed application should behave like a regular Windows application. It
must not require the user to install Python, .NET, OpenVINO, GenAI, or other
runtime dependencies separately. Those files are bundled in the installer
payload:

- OpenVINO and GenAI DLLs come from `dist/windows/ovms`.
- Bundled Python support comes from the `python_on` OVMS package.
- The tray app is published self-contained so it does not require a separate
  .NET Desktop Runtime install.
- Inno Setup is a build-time tool only; users do not need it installed.

The full Manager UI is planned as a later phase.

## Build Order

```powershell
.\windows_build.bat --with_python
.\windows_create_package.bat opt --with_python
.\packaging\windows\manager\build_manager.ps1
.\packaging\windows\installer\build_installer.ps1
```

Expected output:

```text
dist/windows/OVMS-Setup.exe
```
