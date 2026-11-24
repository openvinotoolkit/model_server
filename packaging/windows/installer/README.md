# OVMS Windows Installer

The installer is built with Inno Setup from `OVMSInstaller.iss`.

It expects the OVMS portable package at:

```text
dist/windows/ovms
```

The installer provides:

- OVMS files under `C:\Program Files\OVMS`.
- Per-user data under `%LOCALAPPDATA%\OVMS`.
- Three Python modes: system Python, embedded Python, or no Python.
- Optional OVMS PATH configuration.
- Optional Windows service installation and service control.
- Repair and uninstall actions.
- Start Menu server-control shortcuts.

The installer is fully offline. All runtime dependencies required by the package must be present in `dist/windows/ovms` before building.
