"""Prepare standalone C++ minja round-trip runs using the pinned SDK."""
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
scratch = Path(r'C:\gm\lfm-2026.4\literal-tests')
sdk = Path(r'C:\gm\qwenmonster\semantic-refit-2026.4-sdk\extracted\openvino_genai_windows_2026.4.0.0rc1_x86_64\runtime')
gtest = Path(r'C:\r20264\msnbck67\external\com_google_googletest\googletest')
rapidjson = Path(r'C:\r20264\msnbck67\external\com_github_tencent_rapidjson\include')
runtime = Path(r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\extracted-py_off\ovms')
for mode in ('red', 'green'):
    flag = '/DLFM_TEMPLATE_ORIGINAL' if mode == 'red' else ''
    lines = ['@echo off', 'call C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat >nul', f'cd /d {scratch}',
             f'cl /nologo /EHsc /std:c++17 /utf-8 /MD /DLFM_TEMPLATE_STANDALONE {flag} /I"{repo}" /I"{sdk / "include"}" /I"{rapidjson}" /I"{gtest / "include"}" /I"{gtest}" "{repo / "src/test/llm/output_parsers/lfm_template_roundtrip_test.cpp"}" "{gtest / "src/gtest-all.cc"}" "{gtest / "src/gtest_main.cc"}" "{sdk / "lib/intel64/Release/openvino.lib"}" "{sdk / "lib/intel64/Release/openvino_genai.lib"}" /Fe:template-{mode}.exe > "{repo / "docs/lfm/evidence" / ("template-" + mode + "-compile.log")}" 2>&1',
             'if errorlevel 1 exit /b 2', f'set PATH={runtime};C:\\Windows\\System32;C:\\Windows', f'cd /d {repo}',
             f'{scratch / ("template-" + mode + ".exe")} --gtest_output=xml:{repo / "docs/lfm/evidence" / ("template-" + mode + ".xml")} > "{repo / "docs/lfm/evidence" / ("template-" + mode + ".log")}" 2>&1', 'exit /b %errorlevel%']
    (scratch / ('run-template-' + mode + '.cmd')).write_text('\n'.join(lines) + '\n')
print(scratch)
