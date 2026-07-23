## Python Runtime Setup and Troubleshooting {#ovms_docs_python_runtime_setup_and_troubleshooting}

This document covers Python-related setup, fallback behavior, and common runtime issues for baremetal OVMS deployments.

## Scope

Use this guide when:
- You use a package with Python support.
- You need Python nodes.
- You need advanced LLM chat-template behavior.
- You see Python runtime or plugin errors during startup.

For installation steps of baremetal packages, see [Deploying Model Server on Baremetal](deploying_server_baremetal.md).

## Python Support Modes

Model Server supports two deployment configurations:

**With Python Support** (`PYTHON_DISABLE=0`, default):
- Python nodes are available.
- LLM chat-template handling has broader feature coverage.
- Runtime Python libraries must be available.

**Without Python Support** (`PYTHON_DISABLE=1`):
- Lightweight deployment for C++ models only.
- Python nodes cannot be used.
- LLM template rendering has limitations.
- No Python runtime dependencies are required.

## Graceful Degradation

With Python support enabled, OVMS can continue serving requests when some Python dependencies are missing.

Behavior:
1. If Python interpreter initialization fails, server startup continues with Python features disabled.
2. If Python backend/module loading fails, server remains running without Python features.
3. If Python calculators plugin fails to load, only Python calculators/nodes are unavailable.
4. Non-Python models and graphs continue to work.

## Runtime Setup Checklist

### Linux

- Ensure `LD_LIBRARY_PATH` points to package libraries:

```bash
export LD_LIBRARY_PATH=${PWD}/ovms/lib
```

- Ensure `PYTHONPATH` points to OVMS Python package:

```bash
export PYTHONPATH=${PWD}/ovms/lib/python
```

- Install template dependencies:

```bash
pip3 install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
```

- If using Python nodes with OpenVINO/OpenVINO GenAI, install NumPy:

```bash
pip3 install numpy
```

Do not install `openvino`, `openvino-tokenizers`, or `openvino-genai` via pip for OVMS Python package usage.

### Windows

Run setup in the same shell before starting OVMS:

```bat
.\ovms\setupvars.bat
```

or in PowerShell:

```powershell
.\ovms\setupvars.ps1
```

Optional explicit `PYTHONPATH`:

Command Prompt:

```bat
set PYTHONPATH=%CD%\ovms\lib\python
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\ovms\lib\python"
```

## Common Issues

### Error: "Failed to initialize Python interpreter"

Cause:
- System Python shared library is missing or unavailable in runtime environment.

Fix:
- Ubuntu: `sudo apt install libpython3.12-dev`
- RHEL: `sudo yum install python312-devel`
- Windows: rerun `setupvars.bat` or `setupvars.ps1` in the same shell before starting `ovms`

Check:

```bash
find /usr -name "libpython*" -type f
```

### Error: "Failed to create Python backend"

Cause:
- `pyovms` module cannot be imported.
- `PYTHONPATH` is missing or incorrect.

Fix:

```bash
export PYTHONPATH=${PWD}/ovms/lib/python
```

Check:

```bash
python3 -c "import pyovms; print(pyovms.__file__)"
```

### Warning: "Python calculators plugin failed to load"

Cause:
- Optional calculators plugin library is missing from library search path.

Impact:
- Python nodes/calculators are unavailable.
- Server stays up and non-Python models keep working.

Fix:
- Ensure `libpython_calculators.so` (Linux) or `libpython_calculators.dll` (Windows) is available in runtime library paths.

## Verify Python Feature Availability

1. Start OVMS and inspect logs for Python runtime/plugin initialization messages.
2. Load a Python node graph and confirm it initializes.
3. Confirm package files exist:
   - `${PWD}/ovms/lib/python` for Python package files
   - `${PWD}/ovms/lib` for runtime libraries

## Migration Between Package Variants

**From without-Python to with-Python**:
1. Download with-Python package.
2. Extract to same location.
3. Repeat OS-specific setup (`PYTHONPATH`, dependencies, or `setupvars`).
4. Restart server.

**From with-Python to without-Python**:
1. Download without-Python package.
2. Extract to same location.
3. Repeat OS-specific setup for selected package variant.
4. Restart server.
