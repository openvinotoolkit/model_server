## Python Runtime Setup and Troubleshooting {#ovms_docs_python_runtime_setup_and_troubleshooting}

This document covers Python-related setup, fallback behavior, and common runtime issues for baremetal OVMS deployments.

## Scope

Use this guide when:
- You want to enable optional Python integration.
- You need Python nodes.
- You need advanced LLM chat-template behavior.
- You see Python runtime or plugin errors during startup.

For installation steps of baremetal packages, see [Deploying Model Server on Baremetal](deploying_server_baremetal.md).

## Python Support Modes

Model Server provides one package per operating system. There are no separate
`python_on` and `python_off` downloads. Python integration depends on the
availability of a compatible Python installation at runtime.
The Windows package includes an embedded Python runtime with Jinja2 and
MarkupSafe in `ovms\python`; no separate installation of Python or these packages
is required. This subfolder can be deleted if neither Python feature is needed.

**With Python integration available**:
- A Jinja-based chat-template engine is available for LLM models.
- Python custom nodes can run in MediaPipe graphs.
- Runtime Python libraries must be available.

**Without Python integration available**:
- The same package can serve models that do not require Python.
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

1. Keep the extracted package directory structure intact.
2. On Windows, keep the embedded runtime in `ovms\python`. On Linux, install
   Python 3.12 for Ubuntu 24.04 and Red Hat Enterprise Linux 9, or Python 3.14
   for Ubuntu 26.04.
3. On Linux, install Jinja2 for that interpreter; MarkupSafe is installed as a
   dependency. Both packages are already included in the Windows embedded runtime.
4. Install NumPy if your Python nodes use OpenVINO or OpenVINO GenAI.
5. Restart OVMS after installing dependencies.

For OS-specific commands, see [Optional Python Integration](deploying_server_baremetal.md#optional-python-integration).
No `LD_LIBRARY_PATH`, `PYTHONPATH`, or `setupvars` configuration is required.

Do not install `openvino`, `openvino-tokenizers`, or `openvino-genai` via pip for
OVMS usage; use the versions included in the package.

## Common Issues

### Error: "Failed to initialize Python interpreter"

Cause:
- The Python runtime is missing or unavailable: a system shared library on
  Linux, or the embedded runtime on Windows.

Fix:
- Ubuntu 24.04: `sudo apt install libpython3.12`
- Ubuntu 26.04: `sudo apt install libpython3.14`
- Red Hat Enterprise Linux 9: `sudo dnf install python3.12-libs`
- Windows: restore the `ovms\python` subfolder by re-extracting the Windows
  package if Python features are needed.

Check:

```text
find /usr -name "libpython*" -type f
```

### Error: "Failed to create Python backend"

Cause:
- `pyovms` module cannot be imported.
- The package is incomplete, its directory structure has changed, or its Python
  extensions do not match the installed Python runtime.

Fix:

- Re-extract the package for your operating system into a clean directory.
- Keep the bundled Python modules and runtime libraries in their original locations.
- Use the compatible Python version listed in the runtime setup checklist.
- Restart OVMS and inspect its Python initialization logs.

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
   - Linux: `ovms/lib/python` for Python package files and `ovms/lib` for runtime libraries.
   - Windows: `ovms\python` for the embedded Python runtime and packages.

## Migrating from Older Package Variants

1. Download the single package for your operating system from the
   [baremetal deployment guide](deploying_server_baremetal.md).
2. Extract it into a clean directory rather than overlaying an older package.
3. Remove shell configuration that points to libraries or Python modules from
   the older package.
4. If Python features are needed, keep the embedded runtime on Windows or
   install the compatible runtime on Linux, and ensure Jinja2 is installed.
5. Restart OVMS using the executable in the new package.
