# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");

"""Standalone Windows installer smoke tests with a composer-style Markdown report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from tests.functional.utils.ports import probe_port
except ImportError:
    from functional.utils.ports import probe_port

ROOT = Path(__file__).resolve().parents[2]
BASE_CASES = ("invalid-mode", "none-components", "system-python", "embedded-python", "components-only")
COMPONENTS = (
    "core",
    "python_system",
    "python_embedded",
    "python_none",
    "espeak",
    "npu",
    "gpu",
    "opencv_mp",
    "onnx_frontend",
    "paddle_frontend",
    "tensorflow_frontend",
)
CASES = BASE_CASES + COMPONENTS

COMPONENT_ARTIFACTS = {
    "core": (
        "ovms.exe",
        "openvino.dll",
        "openvino_genai.dll",
        "openvino_gguf_frontend.dll",
        "libovmspython.dll",
    ),
    "python_system": ("python\\pyovms.pyd",),
    "python_embedded": ("python\\python.exe", "python\\pyovms.pyd"),
    "python_none": (),
    "espeak": ("espeak-ng.dll", "espeak-ng-data"),
    "npu": (
        "openvino_intel_npu_compiler.dll",
        "openvino_intel_npu_compiler_loader.dll",
        "openvino_intel_npu_plugin.dll",
        "openvino_intel_npu_vm_runtime.dll",
    ),
    "gpu": ("openvino_intel_gpu_plugin.dll",),
    "opencv_mp": (
        "opencv_world4140.dll",
        "ovms_opencv_calculators.dll",
        "ovms_opencv_image_decoder.dll",
    ),
    "onnx_frontend": ("openvino_onnx_frontend.dll",),
    "paddle_frontend": ("openvino_paddle_frontend.dll",),
    "tensorflow_frontend": (
        "openvino_tensorflow_frontend.dll",
        "openvino_tensorflow_lite_frontend.dll",
    ),
}


def free_port() -> int:
    port = probe_port(0, "127.0.0.1")
    if port is None:
        raise RuntimeError("could not find a free loopback port")
    return port


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)


def find_installer(path: Path) -> Path:
    installer = path.resolve()
    if not installer.is_file():
        raise RuntimeError(f"installer not found: {installer}")
    return installer


def discover_python314() -> Path | None:
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Python314" / "python.exe",
        Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "Python314" / "python.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python314" / "python.exe",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def wait_ready(url: str, process: subprocess.Popen, timeout: int = 45) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OVMS exited before readiness; exit={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as error:
            last_error = str(error)
        time.sleep(1)
    raise RuntimeError(f"OVMS readiness timeout: {last_error}")


def start_and_check(install_dir: Path) -> str:
    ovms = install_dir / "ovms.exe"
    if not ovms.is_file():
        raise RuntimeError(f"missing OVMS executable: {ovms}")

    work = install_dir.parent / "ovms-smoke-data"
    work.mkdir(parents=True, exist_ok=True)
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    config = work / "config.json"
    config.write_text('{"model_config_list":[]}', encoding="utf-8")

    rest_port = free_port()
    grpc_port = free_port()
    while grpc_port == rest_port:
        grpc_port = free_port()

    log_path = work / "ovms.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(ovms), "--config_path", str(config), "--rest_port", str(rest_port),
             "--port", str(grpc_port), "--log_level", "DEBUG"],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(install_dir),
        )
    try:
        wait_ready(f"http://127.0.0.1:{rest_port}/v2", process)
        return f"OVMS readiness passed on REST {rest_port}/gRPC {grpc_port}"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8", errors="replace")


def read_install_options() -> dict:
    path = Path(os.environ.get("LOCALAPPDATA", "")) / "OVMS" / "install-options.json"
    if not path.is_file():
        raise RuntimeError(f"missing install-options.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def uninstall(install_dir: Path) -> None:
    uninstaller = install_dir / "unins000.exe"
    if uninstaller.is_file():
        run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"])


def verify_component_artifacts(install_dir: Path, artifacts: tuple[str, ...]) -> str:
    missing = []
    for relative_path in artifacts:
        path = install_dir / relative_path
        if not path.exists():
            missing.append(relative_path)
    if missing:
        raise RuntimeError("missing component artifacts: " + ", ".join(missing))
    if artifacts:
        return "Verified " + ", ".join(artifacts)
    return "Core-only component state verified"


def install_component(installer: Path, work_root: Path, python_mode: str,
                      components: str, artifacts: tuple[str, ...]) -> Path:
    install_dir = work_root / "install"
    log_path = work_root / "setup.log"
    install_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        f'/DIR={install_dir}', f'/LOG={log_path}',
        f'/COMPONENTS={components}', f'/PYTHONMODE={python_mode}',
    ]
    result = run(command)
    if result.returncode:
        raise RuntimeError(f"installer exit={result.returncode}; {result.stdout[-2000:]}")

    options = read_install_options()
    if options.get("pythonMode") != python_mode:
        raise RuntimeError(f"pythonMode was {options.get('pythonMode')!r}, expected {python_mode!r}")

    if python_mode == "system":
        binding = install_dir / "python" / "pyovms.pyd"
        if not binding.is_file():
            raise RuntimeError(f"missing system-Python binding: {binding}")
        if (install_dir / "python" / "python.exe").is_file():
            raise RuntimeError("embedded python.exe unexpectedly installed in system-Python mode")
    elif python_mode == "embedded":
        if not (install_dir / "python" / "python.exe").is_file():
            raise RuntimeError("missing embedded python.exe")
        if not (install_dir / "python" / "pyovms.pyd").is_file():
            raise RuntimeError("missing embedded pyovms.pyd")
    elif python_mode == "none":
        python_exe = install_dir / "python" / "python.exe"
        if python_exe.is_file():
            raise RuntimeError("embedded python.exe unexpectedly installed in none-Python mode")

    verify_component_artifacts(install_dir, artifacts)
    return install_dir


def run_successful_install(installer: Path, work_root: Path, python_mode: str,
                          components: str, artifacts: tuple[str, ...] = ()) -> tuple[str, str]:
    install_dir = install_component(installer, work_root, python_mode, components, artifacts)
    try:
        readiness = start_and_check(install_dir)
        level1 = verify_component_artifacts(install_dir, artifacts)
        level2 = readiness
        return level1, level2
    finally:
        uninstall(install_dir)


def check_case(case: str, installer: Path) -> list[tuple[str, bool, str]]:
    try:
        with tempfile.TemporaryDirectory(prefix="ovms-installer-", dir=os.environ.get("TEMP")) as temp:
            work_root = Path(temp)
            if case == "invalid-mode":
                log_path = work_root / "invalid.log"
                result = run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                              "/PYTHONMODE=invalid", f"/LOG={log_path}"], timeout=120)
                if result.returncode == 0:
                    return [("", False, "installer accepted invalid /PYTHONMODE")]
                log_candidates = [log_path, installer.parent / "OVMSSetup.log"]
                log_text = "\n".join(read_text(candidate) for candidate in log_candidates if candidate.is_file())
                if "Invalid /PYTHONMODE value" not in log_text:
                    return [("", False, "installer rejection did not contain the expected validation message")]
                return [("", True, "Invalid /PYTHONMODE was rejected by the installer")]

            if case == "none-components":
                level1, level2 = run_successful_install(
                    installer, work_root, "none",
                    "python_none,gpu,opencv_mp,onnx_frontend",
                    ("ovms.exe", "openvino_intel_gpu_plugin.dll", "opencv_world4140.dll", "openvino_onnx_frontend.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case == "system-python":
                if discover_python314() is None:
                    return [(" Level 1", True, "SKIP-ready: Python 3.14 not found"),
                            (" Level 2", True, "SKIP-ready: Python 3.14 not found")]
                level1, level2 = run_successful_install(
                    installer, work_root, "system", "python_system,gpu",
                    ("ovms.exe", "python\\pyovms.pyd", "openvino_intel_gpu_plugin.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case == "embedded-python":
                level1, level2 = run_successful_install(
                    installer, work_root, "embedded", "python_embedded,onnx_frontend",
                    ("ovms.exe", "python\\python.exe", "python\\pyovms.pyd", "openvino_onnx_frontend.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case == "components-only":
                level1, level2 = run_successful_install(
                    installer, work_root, "none", "python_none,paddle_frontend,tensorflow_frontend",
                    ("ovms.exe", "openvino_paddle_frontend.dll", "openvino_tensorflow_frontend.dll", "openvino_tensorflow_lite_frontend.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case in COMPONENTS:
                if case == "python_system":
                    if discover_python314() is None:
                        return [(" Level 1", True, "SKIP-ready: Python 3.14 not found"),
                                (" Level 2", True, "SKIP-ready: Python 3.14 not found")]
                    python_mode, component_selection = "system", "core,python_system"
                elif case == "python_embedded":
                    python_mode, component_selection = "embedded", "core,python_embedded"
                else:
                    python_mode, component_selection = "none", "core,python_none"
                    if case not in ("core", "python_none"):
                        component_selection += "," + case

                try:
                    install_dir = install_component(
                        installer, work_root, python_mode, component_selection,
                        COMPONENT_ARTIFACTS[case])
                except Exception as error:
                    return [(" Level 1", False, str(error))]

                try:
                    try:
                        level1 = verify_component_artifacts(install_dir, COMPONENT_ARTIFACTS[case])
                    except Exception as error:
                        return [(" Level 1", False, str(error))]

                    try:
                        level2 = start_and_check(install_dir)
                        return [(" Level 1", True, level1), (" Level 2", True, level2)]
                    except Exception as error:
                        return [(" Level 1", True, level1), (" Level 2", False, str(error))]
                finally:
                    uninstall(install_dir)

            return [("", False, f"unknown case: {case}")]
    except Exception as error:
        return [(" Level 1", False, str(error))]


def report_passed(path: Path) -> bool:
    found = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        found = True
        status = line.split("|")[2].strip()
        if status not in ("PASS", "SKIP"):
            return False
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows OVMS installer CLI smoke tests")
    parser.add_argument("--installer", type=Path, default=ROOT / "dist" / "windows" / "OVMS-Setup.exe")
    parser.add_argument("--report", type=Path, default=ROOT / "windows_installer_smoke_report.md")
    parser.add_argument("--case", dest="cases", action="append", choices=CASES,
                        help="Run one case; repeat for multiple cases")
    args = parser.parse_args()

    if os.name != "nt":
        parser.error("the Windows installer smoke tests require Windows")

    try:
        installer = find_installer(args.installer)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    cases = args.cases or list(CASES)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", buffering=1) as report:
        report.write("# Windows Installer Smoke-Test Report\n\n")
        report.write(f"- Installer: `{installer}`\n")
        report.write(f"- Run started: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}\n")
        report.write(f"- Cases: `{' '.join(cases)}`\n\n")
        report.write("| Case | Result | Summary |\n| --- | --- | --- |\n")

        for case in cases:
            print(f"Running installer smoke test: {case}", flush=True)
            results = check_case(case, installer)
            for level, passed, detail in results:
                if detail.startswith("SKIP-ready:"):
                    status = "SKIP"
                    detail = detail.removeprefix("SKIP-ready:").strip()
                else:
                    status = "PASS" if passed else "FAIL"
                summary = " ".join(detail.splitlines()).replace("|", "/")
                label = f"`{case}`{level}"
                print(f"{status}: {case}{level}: {summary}", flush=True)
                report.write(f"| {label} | {status} | {summary} |\n")

    passed = report_passed(args.report)
    verdict = "Overall result: PASS" if passed else "Overall result: FAIL"
    with args.report.open("a", encoding="utf-8") as report:
        report.write(f"\n{verdict}\n")
    print(verdict)
    print(f"Report written to {args.report}")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
