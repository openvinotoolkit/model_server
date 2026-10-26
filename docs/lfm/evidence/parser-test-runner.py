"""Link stream regressions against already built parser libraries without changing the baseline."""
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
scratch = Path(r'C:\gm\lfm-2026.4\literal-tests')
sdk = Path(r'C:\gm\qwenmonster\semantic-refit-2026.4-sdk\extracted\openvino_genai_windows_2026.4.0.0rc1_x86_64\runtime')
gtest = Path(r'C:\r20264\msnbck67\external\com_google_googletest\googletest')
rapidjson = Path(r'C:\r20264\msnbck67\external\com_github_tencent_rapidjson\include')
runtime = Path(r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\extracted-py_off\ovms')
libs = list((repo / 'bazel-bin/src/llm').glob('io_processing_*.lib'))
libs += [repo / 'bazel-bin/src/llm/output_parsers.lib', repo / 'bazel-bin/src/test_platform_utils.lib',
         repo / 'bazel-bin/src/libovmslogging.lib', repo / 'bazel-bin/src/libovmsstring_utils.lib',
         sdk / 'lib/intel64/Release/openvino.lib', sdk / 'lib/intel64/Release/openvino_genai.lib']
libs += [repo / 'bazel-bin/src/libovmsstatus.lib', repo / 'bazel-bin/src/llm/partial_json_builder.lib',
         repo / 'bazel-bin/src/utils/env_guard.lib',
         repo / 'bazel-bin/external/com_github_glog_glog/glog.lib',
         repo / 'bazel-bin/external/com_github_gflags_gflags/gflags.lib']
assert all(p.exists() for p in libs), [str(p) for p in libs if not p.exists()]
response = scratch / 'parser-libs.rsp'
response.write_text('\n'.join('"' + str(p) + '"' for p in libs))
lines = ['@echo off', 'call C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat >nul', f'cd /d {scratch}',
         f'cl /nologo /EHsc /std:c++17 /utf-8 /MD /I"{repo}" /I"{sdk / "include"}" /I"{rapidjson}" /I"{gtest / "include"}" /I"{gtest}" "{repo / "src/test/llm/output_parsers/lfm_strict_output_parser_test.cpp"}" "{gtest / "src/gtest-all.cc"}" "{gtest / "src/gtest_main.cc"}" @{response} /Fe:parser-red.exe > "{repo / "docs/lfm/evidence/parser-red-compile.log"}" 2>&1',
         'if errorlevel 1 exit /b 2', f'set PATH={runtime};C:\\Windows\\System32;C:\\Windows', f'cd /d {repo}',
         f'{scratch / "parser-red.exe"} --gtest_output=xml:{repo / "docs/lfm/evidence/parser-red.xml"} > "{repo / "docs/lfm/evidence/parser-red.log"}" 2>&1', 'exit /b %errorlevel%']
(scratch / 'run-parser-red.cmd').write_text('\n'.join(lines) + '\n')
print('Parser libraries:', len(libs))
