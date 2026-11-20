"""Build and exercise composed OVMS images without loading the pytest harness."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from . import docker_compose_demo_probe as probe
else:
    import docker_compose_demo_probe as probe

ARTIFACTS = {
    "base": (),
    "gpu": ("libopenvino_intel_gpu_plugin.so*",),
    "npu": ("libopenvino_intel_npu_plugin.so*", "libopenvino_intel_npu_compiler.so*"),
    "opencv": ("libopencv*.so*",),
    "python": ("libovmspython.so*",),
    "tensorflow": ("libopenvino_tensorflow_frontend.so*",),
    "paddle": ("libopenvino_paddle_frontend.so*",),
    "onnx": ("libopenvino_onnx_frontend.so*",),
    "espeak": ("libespeak-ng.so*", "/ovms/share/espeak-ng-data/phontab"),
}
SIZES_MB = (0, 500, 300, 50, 95, 5, 2, 5, 10)
ERRORS = (OSError, subprocess.SubprocessError, RuntimeError, ValueError, KeyError, IndexError, TypeError)
MISSING_DEVICE_FAILURES = {
    ("`gpu`", "GPU smoke test requires a host render device under /dev/dri"),
    ("`gpu` Level 2", "GPU Level 2 requires a host render device"),
    ("`npu` Level 2", "missing host NPU device /dev/accel/accel0"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build composed OVMS images and run bounded KServe inference. All mode runs sequentially and reports each result.",
        epilog="Approximate size increases over the Ubuntu 24.04 baseline: "
        + ", ".join(f"{name} +{size} MB" for name, size in zip(ARTIFACTS, SIZES_MB))
        + ". GPU inference needs a host render device; NPU execution requires Level 2 fixtures and NPU hardware.",
    )
    parser.add_argument("full_image", help="Source OVMS Docker image.")
    parser.add_argument("component", choices=[*ARTIFACTS, "all"])
    parser.add_argument("--skip-pull", action="store_true", help="Use the local source image without pulling it.")
    parser.add_argument("--with-python", action="store_true", help="Also retain Python in non-Python images.")
    parser.add_argument("--with-opencv", action="store_true", help="Also retain OpenCV in non-OpenCV images.")
    parser.add_argument("--report", type=Path, default=probe.ROOT / "test_log/docker_compose_smoke_report.md",
                        help="Markdown report path (default: test_log/docker_compose_smoke_report.md).")
    parser.add_argument("--demo", dest="fixtures", action="store_const", const=probe.FIXTURES,
                        help="Run Level 2 using repository fixtures.")
    parser.add_argument("--demo-fixtures", dest="fixtures", type=Path, metavar="DIR",
                        help="Run Level 2 using a custom fixture directory.")
    parser.add_argument("--demo-component", action="append", choices=list(ARTIFACTS)[1:],
                        help="Limit Level 2 to this component (repeatable; requires --demo or --demo-fixtures).")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="Check the GPU image's artifacts on CPU if no render device exists; do not validate GPU inference.")
    args = parser.parse_args()
    if args.demo_component and args.fixtures is None:
        parser.error("--demo-component requires --demo or --demo-fixtures")
    return args


def run_smoke(args, component, image, container, work_dir, device_args):
    features = {component} - {"base"}
    if args.with_python:
        features.add("python")
    if args.with_opencv:
        features.add("opencv")
    flags = [f"--{name}" for name in ARTIFACTS if name in features]
    if args.skip_pull:
        flags.append("--skip-pull")
    print(f"Building {image} from {args.full_image}", flush=True)
    result = subprocess.run([sys.executable, str(probe.ROOT / "docker_compose.py"), "--full-image", args.full_image,
                             "--output-name", image, "--work-dir", str(work_dir), *flags], timeout=900)
    if result.returncode:
        raise RuntimeError(f"composer build failed (exit {result.returncode}); see command output")

    target_device = "GPU" if component == "gpu" and device_args else "CPU"
    server_args = ["--model_path", "/models/dummy.xml", "--model_name", "dummy", "--target_device", target_device]
    mounts = probe.mount(probe.ROOT / "src/test/dummy/1", "/models")
    with probe.server(container, image, device_args + mounts, server_args) as url:
        probe.wait_ready(container, url, "dummy")
        for feature in sorted(features):
            for artifact in ARTIFACTS[feature]:
                pattern = Path(artifact if artifact.startswith("/") else f"/ovms/lib/{artifact}")
                matches = probe.docker("exec", container, "find", str(pattern.parent), "-maxdepth", "1",
                               "-name", pattern.name, "-print", "-quit", check=False)
                if matches.returncode or not matches.stdout.strip():
                    raise RuntimeError(f"missing component artifact: {pattern}")
        body = json.dumps({"inputs": [{"name": "b", "shape": [1, 10], "datatype": "FP32",
                                       "data": list(range(10))}]}).encode()
        response = json.loads(probe.request(f"{url}/v2/models/dummy/infer", 15, body))
        if not response.get("outputs"):
            raise ValueError("dummy inference response did not contain outputs")
        print(json.dumps(response), flush=True)
    detail = "Dummy IR inference passed"
    if features:
        detail += f"; checked artifacts for {', '.join(sorted(features))}"
    if component == "gpu":
        detail += "; targeted GPU" if device_args else "; used CPU because no GPU render device was available"
    if component == "npu":
        detail += "; NPU inference not exercised at Level 1"
    return detail + "."


def component_checks(args, component, work_dir):
    image = f"ovms-compose-smoke:{component}"
    container = f"ovms-compose-smoke-{component}-{os.getpid()}"
    device_args = []
    try:
        if component == "gpu":
            render_node = next((node for node in sorted(Path("/dev/dri").glob("renderD*")) if node.is_char_device()), None)
            if render_node is None and not args.allow_cpu_fallback:
                raise RuntimeError("GPU smoke test requires a host render device under /dev/dri")
            if render_node is not None:
                device_args = ["--device", "/dev/dri", "--group-add", str(render_node.stat().st_gid)]
        yield "", True, run_smoke(args, component, image, container, work_dir, device_args)
    except ERRORS as error:
        yield "", False, str(error)
        return

    if args.fixtures is not None and component != "base" and (not args.demo_component or component in args.demo_component):
        try:
            if component == "gpu" and not device_args:
                raise RuntimeError("GPU Level 2 requires a host render device")
            yield " Level 2", True, probe.run_demo(component, args.fixtures, image, container, work_dir, device_args)
        except ERRORS as error:
            yield " Level 2", False, str(error)


def report_passed(path):
    found_result = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        component, status, detail = (cell.strip() for cell in line.strip("|").split("|"))
        found_result = True
        if status != "PASS" and (status != "FAIL" or (component, detail) not in MISSING_DEVICE_FAILURES):
            return False
    return found_result


def main():
    args = parse_args()
    components = list(ARTIFACTS) if args.component == "all" else [args.component]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", buffering=1) as report:
        report.write(f"# Docker Composer Smoke-Test Report\n\n- Full image: `{args.full_image}`\n"
                     f"- Run started: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}\n"
                     f"- Components: `{' '.join(components)}`\n")
        if args.fixtures is not None:
            report.write(f"- Level 2 fixtures: `{args.fixtures}`\n")
            if args.demo_component:
                report.write(f"- Level 2 components: `{' '.join(args.demo_component)}`\n")
        report.write("\n| Component | Result | Summary |\n| --- | --- | --- |\n")
        for component in components:
            print(f"Running composer smoke test: {component}", flush=True)
            with TemporaryDirectory() as work_dir:
                for level, passed, detail in component_checks(args, component, Path(work_dir)):
                    status = "PASS" if passed else "FAIL"
                    print(f"{status}: {component}{level}: {detail}", flush=True)
                    summary = " ".join(detail.splitlines()).replace("|", "/")
                    report.write(f"| `{component}`{level} | {status} | {summary} |\n")
    passed = report_passed(args.report)
    verdict = f"Overall result: {'PASS' if passed else 'FAIL'}"
    with args.report.open("a", encoding="utf-8") as report:
        report.write(f"\n{verdict}\n")
    print(verdict)
    print(f"Report written to {args.report}")
    return int(not passed)


if __name__ == "__main__":
    sys.exit(main())