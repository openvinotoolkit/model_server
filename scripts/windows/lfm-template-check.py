"""Run LFM template tests against the pinned 2026.4 tokenizer/Minja runtime.

This small standalone target avoids a full server rebuild. It does not start a
model or confer API/runtime acceptance. All build and test output is retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, default=Path(r"C:\gm\lfm-2026.4\tool-use-20261027\template"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve() / args.label
    output.mkdir(parents=True, exist_ok=True)
    if (output / "RESULT.json").exists():
        parser.error("A result already exists for this label; use a new label to retain evidence")
    runtime = Path(r"C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\extracted-py_off\ovms")
    sdk = Path(r"C:\gm\qwenmonster\semantic-refit-2026.4-sdk\extracted\openvino_genai_windows_2026.4.0.0rc1_x86_64\runtime")
    external = Path(r"C:\r20264\msnbck67\external")
    gtest = external / "com_google_googletest/googletest"
    rapidjson = external / "com_github_tencent_rapidjson/include"
    sources = [repo / "src/test/llm/output_parsers/lfm_agent_template_test.cpp",
               repo / "src/test/llm/output_parsers/lfm_template_roundtrip_test.cpp",
               gtest / "src/gtest-all.cc", gtest / "src/gtest_main.cc"]
    includes = [repo, sdk / "include", rapidjson, gtest / "include", gtest]
    libraries = [sdk / "lib/intel64/Release/openvino.lib", sdk / "lib/intel64/Release/openvino_genai.lib"]
    compile_args = ["cl", "/nologo", "/EHsc", "/std:c++17", "/utf-8", "/MD", "/DLFM_TEMPLATE_STANDALONE"]
    compile_args += ["/I" + str(path) for path in includes]
    compile_args += [str(path) for path in sources + libraries]
    compile_args += ["/Fe:" + str(output / "lfm-template.exe")]
    batch = output / "compile.cmd"
    batch.write_text('@echo off\ncall "C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul\n'
                     + subprocess.list2cmdline(compile_args) + '\nexit /b %errorlevel%\n', encoding="utf-8")
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "label": args.label,
              "scope": "standalone real-tokenizer Minja template tests, no model inference",
              "compile_command": compile_args, "inputs": {}}
    tracked = sources + [repo / "docs/lfm/templates/LFM2-24B-A2B-int4-ov.jinja",
                         repo / "docs/lfm/templates/LFM2.5-8B-A1B-int4-ov.jinja"]
    tracked += [runtime / name for name in ("openvino.dll", "openvino_genai.dll", "openvino_tokenizers.dll")]
    for fixture in ("LFM2-2.6B", "LFM2.5-8B-A1B"):
        tracked += [repo / "src/test/llm_testing/LiquidAI" / fixture / name for name in
                    ("openvino_tokenizer.xml", "openvino_tokenizer.bin", "tokenizer_config.json")]
    for path in tracked:
        result["inputs"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (output / "compile.log").open("wb") as log:
        compiled = subprocess.run(["cmd", "/d", "/c", str(batch)], cwd=output, stdout=log, stderr=subprocess.STDOUT)
    result["compile_exit"] = compiled.returncode
    if not compiled.returncode:
        environment = os.environ.copy()
        environment["PATH"] = str(runtime) + os.pathsep + environment["PATH"]
        command = [str(output / "lfm-template.exe"), "--gtest_output=xml:" + str(output / "results.xml")]
        with (output / "tests.log").open("wb") as log:
            tested = subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
        result.update(test_exit=tested.returncode, test_command=command)
    (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "compile_exit": result["compile_exit"], "test_exit": result.get("test_exit")}, indent=2))
    return result.get("test_exit", compiled.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
