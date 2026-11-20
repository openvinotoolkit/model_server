# OVMS Windows Installer and Manager Plan

## Goal

Create a Windows installer and GUI manager that make OpenVINO Model Server
simple to install, configure, run, and maintain on Windows.

The user-facing experience should feel closer to a local AI desktop server app:

- Install OVMS with the required OpenVINO runtime dependencies.
- Include GenAI/tokenizer support for LLM use cases when selected.
- Optionally include Python-enabled OVMS support.
- Configure server settings during installation.
- Provide a GUI and taskbar tray icon for later management.
- Support start-at-login and Windows Service modes.
- Allow repair, upgrade, and optional feature installation later.

The first implementation should live in the `model_server` repository because
the primary product being installed and managed is `ovms.exe`. OpenVINO Runtime
and OpenVINO GenAI are dependencies consumed through the OVMS Windows package.

## Non-Goals

- Do not turn `ovms.exe` itself into a GUI application.
- Do not install unrelated OpenVINO SDK/toolkit components unless they are
  required by the OVMS package.
- Do not mix arbitrary OpenVINO, GenAI, and OVMS versions. The installer and
  manager should keep the installed stack version-matched.
- Do not replace the portable `ovms.zip` package. The installer should exist
  alongside it.

## Recommended Repository Layout

```text
model_server/
  packaging/
    windows/
      OVMS_WINDOWS_INSTALLER_MANAGER_PLAN.md
      README.md
      installer/
        README.md
        build_installer.ps1
        OVMSInstaller.iss
        assets/
        uninstall/
      manager/
        README.md
        build_manager.ps1
        OVMS.Manager.sln
        src/
          OVMS.Manager/
          OVMS.Manager.Core/
        assets/
      scripts/
        configure-ovms.ps1
        install-service.ps1
        uninstall-service.ps1
        uninstall-ovms.ps1
        repair-package.ps1
        validate-install.ps1
      schemas/
        ovms-manager-settings.schema.json
        install-options.schema.json
```

Existing root-level packaging scripts should remain in place:

```text
windows_build.bat
windows_create_package.bat
install_ovms_service.bat
setupvars.bat
setupvars.ps1
```

The Windows installer should consume the existing package output:

```text
dist/windows/ovms/
dist/windows/ovms.zip
```

Final release artifacts should include:

```text
dist/windows/ovms.zip
dist/windows/OVMS-Setup.exe
```

## Product Architecture

Keep the server headless and add a Windows UX layer around it.

```text
OVMS-Setup.exe
  installs files
  writes first-time config
  optionally registers the Windows Service
  installs OVMS Manager

OVMS Manager.exe
  owns the GUI and tray icon
  controls server startup/shutdown
  manages settings, logs, models, and feature repair

ovms.exe
  remains the model server process
```

## Installer Technology

Start with Inno Setup.

Reasons:

- Fast path to a polished single-file installer.
- Supports component selection, custom wizard pages, Start Menu shortcuts,
  uninstall hooks, and scripted service commands.
- Easier to iterate than WiX/MSI for the first product version.

WiX/MSI can be considered later if enterprise/GPO deployment becomes a firm
requirement.

Inno Setup is a build-time dependency only. The generated installer must be a
normal Windows installer that does not require users to install Inno Setup,
Python, .NET, OpenVINO, GenAI, or other runtime packages separately.

Runtime dependency policy:

- OVMS, OpenVINO runtime DLLs, OpenVINO GenAI DLLs, tokenizers, and bundled
  Python come from the packaged `dist/windows/ovms` payload.
- The tray app must be published self-contained and included in the installer
  payload.
- The installer must not download required runtime dependencies during install.
- PowerShell may be used for install-time configuration because it is part of
  supported Windows installations, but it must not fetch external packages.

## Install Locations

Program files:

```text
C:\Program Files\OVMS\
```

Persistent data:

```text
C:\ProgramData\OVMS\
  settings.json
  models\
    config.json
  logs\
    ovms_server.log
  packages\
  downloads\
```

Per-user startup, when using login mode:

```text
HKCU\Software\Microsoft\Windows\CurrentVersion\Run
```

Service configuration, when using service mode:

```text
HKLM\SYSTEM\CurrentControlSet\Services\ovms
```

## Privilege Model

The installer and Manager must support a clear privilege boundary.

### Machine-Wide Install

Default for the first release:

```text
C:\Program Files\OVMS
C:\ProgramData\OVMS
HKLM service registration
Machine PATH, if selected
```

Machine-wide install requires elevation during setup.

Admin-only actions:

- Installing to `C:\Program Files\OVMS`.
- Writing machine-wide PATH entries.
- Registering or deleting the `ovms` Windows Service.
- Changing service start type.
- Writing service command-line configuration.
- Opening firewall rules, if added later.

### Per-User Actions

These should not require elevation:

- Starting Manager at user login through HKCU Run.
- Starting or stopping a Manager-owned `ovms.exe` process.
- Editing user-writable settings where permissions allow it.
- Opening logs, model folders, and diagnostics.

### Manager Elevation Behavior

Manager should run unelevated by default. When the user performs an admin-only
action, Manager should prompt for elevation for that action only.

Manager should not require elevation just to show the tray icon, read status, or
start a user-owned OVMS process.

### Future Per-User Install

A per-user install may be added later:

```text
%LOCALAPPDATA%\Programs\OVMS
%LOCALAPPDATA%\OVMS
HKCU startup only
No Windows Service support unless elevated later
```

This should be treated as a separate install mode, not an accidental fallback.

## Installer Wizard

### Component Selection

Suggested defaults:

```text
[x] OpenVINO Model Server
[x] GenAI / LLM support
[x] Python support
[x] OVMS Manager app
[x] Tray icon
[ ] Windows Service
[ ] Start at boot
[x] Add OVMS to PATH
```

The UI should present OpenVINO and GenAI as required runtime capabilities of
the selected OVMS package, not as unrelated SDK installs.

### Existing Install Detection

Before showing the normal install flow, the installer must check whether OVMS is
already installed.

Detection sources:

- Apps & Features uninstall registry entry for OpenVINO Model Server.
- Existing install marker under `C:\Program Files\OVMS`.
- Existing Manager settings at `C:\ProgramData\OVMS\settings.json`.
- Existing `ovms` Windows Service registration.
- Existing PATH entries pointing to an OVMS install directory.
- Running `OVMS Manager.exe` process.
- Running `ovms.exe` process from a known OVMS install directory.

Install marker:

```text
C:\ProgramData\OVMS\install.json
```

Suggested content:

```json
{
  "productName": "OpenVINO Model Server",
  "installDir": "C:\\Program Files\\OVMS",
  "dataDir": "C:\\ProgramData\\OVMS",
  "version": "2026.2.1",
  "packageVariant": "python_on",
  "installedAtUtc": "2026-06-30T15:00:00Z",
  "installerVersion": "1.0.0"
}
```

If an existing install is detected, the installer should offer:

```text
Repair existing install
Upgrade existing install
Modify installed features
Uninstall
Cancel
```

Rules:

- Do not silently install over an existing OVMS directory.
- Stop Manager and OVMS before repair, upgrade, modify, or uninstall.
- Preserve `C:\ProgramData\OVMS` by default.
- If multiple possible installs are detected, show the paths and require the user
  to choose or cancel.
- If only a running `ovms.exe` is found outside the managed install directory,
  warn about the conflict but do not treat it as the managed install.

### Server Configuration

Collect:

```text
REST port: 8000
gRPC port: 9000
Bind address: 127.0.0.1
Model repository: C:\ProgramData\OVMS\models
Log level: INFO
Log path: C:\ProgramData\OVMS\logs\ovms_server.log
```

### Runtime Mode

Offer:

```text
Run when I sign in with tray icon
Run as Windows Service
Manual only
```

Recommended default:

```text
Run when I sign in with tray icon
```

Advanced option:

```text
Run as Windows Service and optionally start at boot
```

### Optional Initial Model

Offer:

```text
None
Add local model
Pull model after install
```

The MVP can start with `None` only and add model workflows in a later phase.

### Finish Page

Offer:

```text
[x] Launch OVMS Manager
[ ] Start server now
```

## Manager Application

Use .NET WPF or WinUI 3. WPF is the recommended first choice because it is
pragmatic, native-feeling, and works well with tray icons, Windows services,
process control, and file-based settings.

Suggested projects:

```text
OVMS.Manager
  WPF UI and tray integration

OVMS.Manager.Core
  service/process/config/log/package logic
```

### Dashboard

Show:

- Server running/stopped state.
- Runtime mode: process, service, or manual.
- Active REST and gRPC ports.
- Bind address.
- Installed package variant.
- Served model count.
- Latest health check result.

Health check:

```text
GET http://127.0.0.1:<restPort>/v3/models
```

### Server Controls

Support:

- Start server.
- Stop server.
- Restart server.
- Reload status.
- Open logs.
- Open model repository.

In user-login mode, Manager starts and stops `ovms.exe` directly.

In service mode, Manager controls the `ovms` Windows Service.

### Settings

Support editing:

- REST port.
- gRPC port.
- Bind address.
- Log level.
- Log path.
- Model repository path.
- Config path.
- Target device defaults, if added later.
- Startup mode.
- Show tray icon.

Settings changes that affect the process command line should prompt for or
perform a restart.

### Startup

Support:

- Start Manager at user login.
- Show tray icon.
- Start OVMS when Manager starts.
- Register as Windows Service.
- Configure service start type: manual or automatic.

### Models

Support:

- List configured models from `config.json`.
- Add local model.
- Remove model from config.
- Open model folder.
- Refresh/reload config.
- Pull supported model, in a later phase.

### Features

Detect:

- `ovms.exe` exists.
- `openvino_genai.dll` exists.
- `openvino_tokenizers.dll` exists.
- `python\python.exe` exists, for Python-enabled package.
- `ovms.exe --version` succeeds.

Show:

```text
GenAI support: Installed / Missing
Python support: Installed / Missing
Package variant: python_on / python_off
```

If GenAI or Python support is missing, offer repair or upgrade to a matching
package variant rather than mixing arbitrary DLL versions.

### Updates

Support:

- Check for updates manually from Manager.
- Optionally check for updates periodically.
- Show installed version, latest available version, package variant, and release
  notes link.
- Download updates only after user confirmation.
- Reuse the staged upgrade and rollback flow.

The first release should not silently auto-install updates.

### Logs

Support:

- Tail `ovms_server.log`.
- Open log folder.
- Copy diagnostics summary.
- Show recent service events, if practical.

### Advanced

Support:

- View effective server command line.
- Validate PATH and environment variables.
- Reset config.
- Repair package.
- Check for updates.
- Export diagnostics bundle.

## Tray Icon

The tray icon belongs to `OVMS Manager.exe`, not `ovms.exe`.

Left-click behavior:

- Open or focus the OVMS Manager window.
- If the Manager window is minimized or hidden, restore it.
- If the Manager window is already open, bring it to the foreground.

Context menu:

```text
Open OVMS Manager
Start Server
Stop Server
```

Keep the first version intentionally small. Additional items such as restart,
logs, diagnostics, model folder, settings, and quit can be added later from the
Manager UI if needed.

Notifications:

- Server started.
- Server stopped.
- Server failed to start.
- Port already in use.
- Model config changed.

## Configuration Files

Manager settings:

```text
C:\ProgramData\OVMS\settings.json
```

Initial content:

```json
{
  "installDir": "C:\\Program Files\\OVMS",
  "dataDir": "C:\\ProgramData\\OVMS",
  "modelRepositoryPath": "C:\\ProgramData\\OVMS\\models",
  "configPath": "C:\\ProgramData\\OVMS\\models\\config.json",
  "logPath": "C:\\ProgramData\\OVMS\\logs\\ovms_server.log",
  "restPort": 8000,
  "grpcPort": 9000,
  "bindAddress": "127.0.0.1",
  "logLevel": "INFO",
  "runMode": "user-login",
  "startAtLogin": true,
  "showTrayIcon": true,
  "serviceAutoStart": false,
  "packageVariant": "python_on"
}
```

OVMS config:

```text
C:\ProgramData\OVMS\models\config.json
```

Initial content:

```json
{
  "model_config_list": []
}
```

## Startup Modes

### User-Login Mode

Behavior:

- Manager starts at user login.
- Manager owns the tray icon.
- Manager starts `ovms.exe` hidden.
- Server runs only after user login.

This is the best Ollama-like default.

### Service Mode

Behavior:

- `ovms` Windows Service owns the server process.
- Service can start before user login.
- Manager controls service state.
- Tray icon is still provided by Manager after login.

This is best for machine/server reliability.

### Manual Mode

Behavior:

- Nothing starts automatically.
- User launches Manager and starts OVMS manually.

This is best for development and low-impact installs.

## Runtime Ownership

Exactly one runtime owner should control OVMS at a time.

Runtime owners:

```text
manager-process
windows-service
manual
none
```

Manager must avoid starting a duplicate server when:

- The `ovms` Windows Service is running.
- A Manager-owned `ovms.exe` process is already running.
- Another process is already listening on the configured REST or gRPC port.
- A lock file indicates another Manager instance is controlling the server.

State file:

```text
C:\ProgramData\OVMS\runtime.json
```

Suggested content:

```json
{
  "owner": "manager-process",
  "pid": 12345,
  "serviceName": "ovms",
  "restPort": 8000,
  "grpcPort": 9000,
  "startedAtUtc": "2026-06-30T15:00:00Z"
}
```

Lock file:

```text
C:\ProgramData\OVMS\ovms-manager.lock
```

Ownership rules:

- User-login mode may start only a Manager-owned process.
- Service mode may start only the `ovms` Windows Service.
- Manual mode never auto-starts OVMS.
- Switching from process mode to service mode must stop the process first.
- Switching from service mode to process mode must stop the service first.
- Manager should show the current owner and block conflicting start actions with
  a clear explanation.

Port preflight:

- Before starting OVMS, Manager must check configured REST and gRPC ports.
- If a port is occupied by another process, Manager should show the process id
  and executable name when available.
- Manager should not blindly kill unrelated processes.

## Feature and Package Strategy

Use the OVMS Windows package as the unit of installation.

Recommended variants:

```text
Minimal install: OVMS python_off package
LLM/GenAI install: OVMS python_on package
```

Do not independently install unmatched OpenVINO Runtime, GenAI, and tokenizer
DLLs. The Manager should install, repair, or upgrade a matching OVMS package
variant.

This avoids version mismatch issues between:

- `ovms.exe`
- OpenVINO runtime DLLs
- OpenVINO GenAI DLLs
- OpenVINO tokenizers DLLs
- Python support files

## Upgrade and Repair Strategy

Upgrade and repair must be staged and recoverable.

Package source:

- Installer builds from local `dist/windows/ovms`.
- Manager repair/upgrade uses a matching OVMS release package or an installer
  cache under `C:\ProgramData\OVMS\packages`.

Verification before use:

- Validate package version metadata.
- Validate expected files are present.
- Validate checksums or signatures when packages are downloaded.
- Run `ovms.exe --version` from the staged package before swapping.

Staging path:

```text
C:\ProgramData\OVMS\packages\staging\<version>\
```

Backup path:

```text
C:\ProgramData\OVMS\packages\backup\<version>\
```

Upgrade steps:

1. Stop Manager-owned OVMS process or `ovms` service.
2. Download or locate the matching package.
3. Verify checksum/signature and required files.
4. Extract to staging.
5. Run staged `ovms.exe --version`.
6. Backup current install files.
7. Swap staged files into `C:\Program Files\OVMS`.
8. Preserve `C:\ProgramData\OVMS` settings, models, and logs.
9. Restart OVMS if it was running before upgrade.
10. Run health check.
11. Roll back from backup if version or health validation fails.

Repair steps:

- Re-verify installed files.
- Restore missing files from package cache when possible.
- Re-run `ovms.exe --version`.
- Re-register service only if the selected runtime mode requires it.

The Manager should never mix individual DLLs from unrelated package versions.

## Update Checking

Update checking should be explicit, version-aware, and safe.

Default behavior:

- Manual update checks are supported.
- Periodic update checks are opt-in.
- Updates are downloaded and installed only after user confirmation.

Update source:

- Official OVMS release metadata.
- For development builds, an explicitly configured package feed may be used.

The Manager should compare:

- Installed OVMS version.
- Installed package variant, such as `python_on` or `python_off`.
- Installed Manager version.
- Latest compatible OVMS package.
- Latest compatible Manager package.

Update states:

```text
Up to date
Update available
Update downloaded
Update failed
Update requires restart
Cannot check for updates
```

Update flow:

1. User clicks "Check for updates".
2. Manager fetches release metadata.
3. Manager shows latest compatible version and release notes link.
4. User confirms download.
5. Manager downloads package to `C:\ProgramData\OVMS\packages`.
6. Manager verifies checksum/signature.
7. Manager stages package.
8. Manager stops OVMS if needed.
9. Manager applies upgrade using the staged upgrade flow.
10. Manager restarts OVMS if it was running before update.
11. Manager reports success or rollback result.

The update checker must never install a package whose version or variant cannot
be matched to the installed stack.

## Configuration Safety

All configuration writes must be validated and recoverable.

Files controlled by Manager:

```text
C:\ProgramData\OVMS\settings.json
C:\ProgramData\OVMS\models\config.json
C:\ProgramData\OVMS\runtime.json
```

Rules:

- Validate JSON before saving.
- Validate against schema where a schema exists.
- Write to a temporary file first.
- Atomically replace the destination file.
- Keep a `.bak` copy of the previous valid version.
- Preserve comments only if a comment-preserving parser is adopted; otherwise
  treat files as strict JSON.
- If config is malformed on startup, show the error and offer restore from
  backup.

Suggested backup names:

```text
settings.json.bak
config.json.bak
runtime.json.bak
```

Restart behavior:

- Settings that alter the OVMS command line require restart.
- Settings that only affect Manager UI should apply immediately.
- Manager must validate config before restarting OVMS.

## Security and Network Exposure

Default bind address must be local-only:

```text
127.0.0.1
```

Binding to these addresses is security-sensitive:

```text
0.0.0.0
:: 
LAN adapter IPs
```

If the user selects a non-local bind address, Manager should:

- Show an explicit warning that the model server may be reachable from other
  devices.
- Require confirmation.
- Show the active URLs after restart.
- Avoid adding firewall rules unless the user explicitly opts in.

If firewall rule management is added later, it must be treated as an admin-only
operation and must be removed during uninstall if created by the installer or
Manager.

## Model Download and Trust

Model download can be added after the MVP, but the plan must account for trust
and storage rules.

Model download requirements:

- Show source, model id, license link, and estimated size before download.
- Support cancellation.
- Support resume when practical.
- Check free disk space before download.
- Use a cache under `C:\ProgramData\OVMS\downloads` by default.
- Support Hugging Face authentication for gated models if needed.
- Do not auto-accept model licenses on behalf of the user.
- Record downloaded model metadata.

Metadata path:

```text
C:\ProgramData\OVMS\models\model-metadata.json
```

Suggested metadata:

```json
{
  "models": [
    {
      "name": "OpenVINO/Qwen3-8B-int4-ov",
      "source": "huggingface",
      "path": "C:\\ProgramData\\OVMS\\models\\OpenVINO\\Qwen3-8B-int4-ov",
      "downloadedAtUtc": "2026-06-30T15:00:00Z",
      "licenseAccepted": false
    }
  ]
}
```

## Installer Responsibilities

Install:

- Copy `dist/windows/ovms` to install directory.
- Copy `OVMS Manager.exe` and supporting files.
- Create `C:\ProgramData\OVMS`.
- Create `install.json`.
- Create default `settings.json`.
- Create default `models\config.json`.
- Add PATH entries if selected.
- Register service if selected.
- Add login startup entry if selected.
- Create Start Menu shortcuts.
- Launch Manager if selected.

Uninstall:

- Stop Manager-controlled process if running.
- Stop service if installed.
- Delete service if installed.
- Remove login startup entry.
- Remove PATH entries created by installer.
- Remove program files.
- Ask whether to preserve models, logs, and settings under `C:\ProgramData\OVMS`.

## Manager Responsibilities

- Read and write `settings.json`.
- Read `install.json` and report installed version/variant.
- Read and update OVMS `config.json`.
- Build the effective `ovms.exe` command line.
- Start and stop `ovms.exe` in user-login/manual modes.
- Start and stop the `ovms` Windows Service in service mode.
- Update service command line when settings change.
- Run health checks.
- Show logs.
- Detect package capabilities.
- Repair or upgrade the installed package.
- Check for updates.
- Manage startup mode.
- Enforce runtime ownership rules.
- Validate ports before starting OVMS.
- Validate and atomically write configuration.
- Export diagnostics bundles.

## Build Pipeline

Initial local pipeline:

```powershell
.\windows_build.bat --with_python
.\windows_create_package.bat opt --with_python
.\packaging\windows\manager\build_manager.ps1
.\packaging\windows\installer\build_installer.ps1
```

Expected outputs:

```text
dist/windows/ovms.zip
dist/windows/OVMS-Setup.exe
```

The installer build should fail clearly if `dist/windows/ovms/ovms.exe` is
missing.

## Uninstaller

Uninstall support is a first-class deliverable. The Windows installer must
register a standard uninstaller in Windows Apps & Features and provide clean
scripted cleanup for files, service state, startup entries, and environment
changes.

Uninstaller entry:

```text
Settings > Apps > Installed apps > OpenVINO Model Server
```

Uninstaller responsibilities:

- Stop `OVMS Manager.exe` if it is running.
- Stop any Manager-launched `ovms.exe` process.
- Stop the `ovms` Windows Service if installed.
- Delete the `ovms` Windows Service if installed.
- Remove login startup registry entries created by the installer.
- Remove PATH entries created by the installer.
- Remove Start Menu and desktop shortcuts.
- Remove installed program files under `C:\Program Files\OVMS`.
- Ask whether to preserve user data under `C:\ProgramData\OVMS`.
- Preserve models, logs, and settings by default unless the user selects full
  removal.

User data removal options:

```text
Preserve models, logs, and settings
Remove logs and settings but keep models
Remove everything, including models
```

The uninstaller should avoid deleting arbitrary user-selected directories unless
they are inside `C:\ProgramData\OVMS` or were explicitly created by the
installer. If the user configured an external model repository, uninstall should
preserve it by default and show the path before offering deletion.

## Diagnostics

Diagnostics should be available before release hardening. The Manager should
include a "Copy diagnostics" or "Export diagnostics bundle" action.

Diagnostics bundle contents:

- Manager version.
- OVMS version from `ovms.exe --version`.
- Installed package variant.
- Install directory.
- Data directory.
- Effective command line with secrets redacted.
- `settings.json`.
- OVMS `config.json`.
- `runtime.json`.
- Service state from `sc query ovms`, when available.
- Service configuration from `sc qc ovms`, when available.
- Port ownership for configured REST and gRPC ports.
- Last 500 lines of `ovms_server.log`, if present.
- Recent Manager log entries.
- Latest health check result.

Diagnostics should be written to:

```text
C:\ProgramData\OVMS\diagnostics\ovms-diagnostics-<timestamp>.zip
```

Sensitive values, tokens, and credentials must be redacted.

## Test Matrix

The Windows installer and Manager should have explicit install-time and
runtime smoke tests.

Installer tests:

- Detect existing install and show repair/upgrade/modify/uninstall choices.
- Clean machine-wide install.
- Install with PATH enabled.
- Install without PATH.
- Install with Manager startup enabled.
- Install with service mode enabled.
- Install to a path containing spaces.
- Install when `C:\ProgramData\OVMS` already exists.
- Install over a previous version.
- Detect existing unmanaged `ovms.exe` and warn without modifying it.

Manager tests:

- Start/stop/restart in user-login mode.
- Start/stop/restart in service mode.
- Switch from process mode to service mode.
- Switch from service mode to process mode.
- Block duplicate start when port is occupied.
- Report owner when another OVMS process is running.
- Persist settings changes.
- Restore malformed config from backup.
- Export diagnostics bundle.

Uninstaller tests:

- Uninstall after process-mode install.
- Uninstall after service-mode install.
- Preserve models/logs/settings.
- Remove logs/settings but keep models.
- Full removal.
- External model repository is preserved by default.
- PATH and startup entries are removed.
- Service is stopped and deleted.

Upgrade/repair tests:

- Repair missing DLL.
- Repair missing Manager file.
- Upgrade python_off to python_on.
- Failed upgrade rolls back.
- Upgrade preserves settings and model repository.
- Update checker reports up-to-date state.
- Update checker reports update-available state.
- Update checker handles offline/network failure.

Security tests:

- Default bind is `127.0.0.1`.
- Non-local bind shows warning.
- Firewall changes require explicit opt-in if implemented.

## Implementation Phases

### Phase 1: Foundation

- Add folder layout.
- Add README files.
- Add settings schema.
- Add install-options schema.
- Add install marker schema.
- Add runtime state schema.
- Add validation script.
- Document service/process/startup behavior.
- Document privilege model.
- Document runtime ownership rules.
- Document config safety rules.

Acceptance criteria:

- `packaging/windows` documents the intended architecture.
- `validate-install.ps1` can inspect an unpacked OVMS folder and report missing
  required files.
- The plan defines admin-only actions and non-admin Manager actions.
- The plan defines duplicate-start prevention.

### Phase 2: Installer MVP

- Add Inno Setup script.
- Add existing-install detection.
- Install existing `dist/windows/ovms` package.
- Write `settings.json`.
- Write empty OVMS `config.json`.
- Create Start Menu shortcuts.
- Support Add to PATH.
- Support optional service registration.
- Register a standard Apps & Features uninstaller.
- Support clean uninstall with preserve/remove data options.
- Implement elevated machine-wide install.
- Fail clearly when required elevation is unavailable.

Acceptance criteria:

- Fresh install succeeds on Windows.
- Existing install shows repair/upgrade/modify/uninstall choices.
- `ovms.exe --version` works after install.
- User can start OVMS manually from installed files.
- Uninstall removes program files, service registration, startup entries, PATH
  changes, and shortcuts.
- Uninstall preserves or removes data according to user choice.
- Installer-created PATH and startup entries are tracked for cleanup.

### Phase 3: Manager MVP

- Add WPF Manager shell.
- Add tray icon.
- Add dashboard.
- Add start/stop/restart controls.
- Add settings page.
- Add logs page.
- Add start-at-login toggle.
- Add runtime ownership detection.
- Add port preflight.
- Add minimal diagnostics export.

Acceptance criteria:

- Manager starts from Start Menu.
- Tray menu works.
- Manager can start and stop `ovms.exe`.
- Health check reflects the running server state.
- Settings changes persist.
- Manager blocks duplicate starts and reports port conflicts.
- Diagnostics bundle can be exported.

### Phase 4: Service Mode

- Manager can register/unregister service.
- Manager can start/stop/restart service.
- Manager can set service start type.
- Installer and Manager use compatible service configuration.
- Switching runtime modes stops the previous owner first.

Acceptance criteria:

- `sc query ovms` reflects selected state.
- Service mode starts OVMS with configured port, bind address, config path, and
  log path.
- Manager reports service errors clearly.
- Manager does not allow process mode and service mode to run OVMS
  simultaneously.

### Phase 5: Model Management

- List configured models.
- Add local model to `config.json`.
- Remove model from `config.json`.
- Open model repository.
- Refresh health/model list.
- Validate config before writing.
- Backup config before writing.

Acceptance criteria:

- Adding a local model updates `config.json`.
- Removing a model updates `config.json`.
- Manager shows served models when OVMS is running.
- Malformed config can be restored from backup.

### Phase 6: Feature Management

- Detect GenAI support.
- Detect Python support.
- Detect package variant.
- Offer repair install.
- Offer upgrade from python_off to python_on using a matching package.
- Add manual update checking.
- Stage package upgrades.
- Verify package before swap.
- Roll back failed upgrades.

Acceptance criteria:

- Manager reports installed capabilities accurately.
- Repair verifies required files after completion.
- Upgrade preserves settings and model repository.
- Failed upgrade restores the previous working package.
- Update checker reports current state accurately.

### Phase 7: Release Hardening

- Add code signing path.
- Add installer versioning.
- Add upgrade behavior.
- Add modify/repair behavior for existing installs.
- Add update checking and package-feed compatibility rules.
- Add diagnostics bundle export.
- Add CI packaging job.
- Add documentation for support and troubleshooting.
- Add automated installer, Manager, service, uninstall, and upgrade smoke tests.

Acceptance criteria:

- Signed installer builds reproducibly.
- Installing over a previous version upgrades cleanly.
- Diagnostics bundle contains settings, logs, version, service state, and health
  check output.
- Test matrix runs successfully on a clean Windows worker.

## Open Questions

- Should the default REST port be 8000, or should it preserve the current local
  AgentTools convention when used there?
- Should the installer default to user-login mode or service mode for non-admin
  installs?
- Should the first version support model download during install, or should that
  be Manager-only after install?
- Should we ship both python_on and python_off installers, or one installer that
  can download the other package variant?
- Should Manager require admin elevation only for service/PATH changes, or run
  elevated at startup?
- Should the first release support per-user install, or only elevated
  machine-wide install?
- What package signature/hash source should Manager trust for repair and
  upgrade downloads?
- Should periodic update checks be opt-in during install or only configurable
  later in Manager?
- What registry keys should be authoritative for installed-version detection?
- Should non-local bind addresses be allowed in the installer, or only in
  Manager advanced settings?
- How should Manager store Hugging Face tokens if gated model downloads are
  supported?

## First Concrete Tasks

1. Add `packaging/windows/README.md` summarizing the installer and manager
   ownership.
2. Add schemas for `settings.json`, install options, and install marker.
3. Add existing-install detection rules.
4. Add runtime state schema and duplicate-start ownership rules.
5. Add `validate-install.ps1` for unpacked `dist/windows/ovms`.
6. Add an Inno Setup MVP that installs the existing package.
7. Add explicit uninstaller scripts and Apps & Features registration.
8. Add a minimal WPF Manager with tray icon and start/stop controls.
9. Add diagnostics export and port preflight.
10. Add manual update checker design and package metadata contract.
