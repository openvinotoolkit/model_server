# OVMS Windows Installer

The installer is built with Inno Setup from `OVMSInstaller.iss`.

It expects the OVMS portable package to already exist at:

```text
dist/windows/ovms
```

The installer adds:

- OVMS files under `C:\Program Files\OVMS`.
- Data folders under `C:\ProgramData\OVMS`.
- Default settings and install marker files.
- Start Menu shortcuts.
- Optional PATH entry.
- Optional start-at-login entry for the tray app.
- Optional Windows Service registration.
- A standard Apps & Features uninstaller.

The installer must be fully offline once built. It should not download Python,
.NET, OpenVINO, GenAI, or model-server dependencies during installation. The
OVMS package and self-contained tray app must already be present in the
installer payload.

The uninstaller stops managed OVMS processes, removes optional service/startup
configuration, removes installer-created PATH entries, and removes installed
program files. Data under `C:\ProgramData\OVMS` is preserved by default.
