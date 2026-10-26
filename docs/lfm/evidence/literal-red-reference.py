"""Extract exact legacy normalizers for the standalone regression RED run."""
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
scratch = Path(r'C:\gm\lfm-2026.4\literal-tests')
scratch.mkdir(parents=True, exist_ok=True)
source = subprocess.check_output(['git', '-C', str(repo), 'show', 'a5f60ccd:src/llm/io_processing/lfm2/lfm2_tool_parser.cpp'], text=True)
functions = source[source.index('std::string parseArrayParameter'):source.index('void writeArgumentToWriter')]
header = '''#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#define SPDLOG_LOGGER_TRACE(...) ((void)0)
namespace ovms::lfm2 {
inline void trim(std::string& s) { const auto b=s.find_first_not_of(" "); const auto e=s.find_last_not_of(" "); s=b==std::string::npos ? "" : s.substr(b,e-b+1); }
'''
header += functions
header += '''inline std::string parseArguments(std::string_view input) {
    return "{\\\"arg\\\":" + normalizeArgStr(std::string(input.substr(4))) + "}";
}
}
'''
(scratch / 'legacy_lfm_normalize.hpp').write_text(header)
gtest = Path(r'C:\r20264\msnbck67\external\com_google_googletest\googletest')
rapidjson = Path(r'C:\r20264\msnbck67\external\com_github_tencent_rapidjson\include')
for mode in ('red', 'green'):
    flag = '/DLFM_LEGACY_REFERENCE' if mode == 'red' else ''
    lines = ['@echo off', 'call C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat >nul',
             f'cd /d {scratch}',
             f'cl /nologo /EHsc /std:c++17 /utf-8 /MD /DLFM_LITERAL_STANDALONE {flag} /I"{repo}" /I"{scratch}" /I"{rapidjson}" /I"{gtest / "include"}" /I"{gtest}" "{repo / "src/test/llm/output_parsers/lfm_literal_test.cpp"}" "{gtest / "src/gtest-all.cc"}" "{gtest / "src/gtest_main.cc"}" /Fe:literal-{mode}.exe > "{repo / "docs/lfm/evidence" / ("literal-" + mode + "-compile.log")}" 2>&1',
             'if errorlevel 1 exit /b 2',
             f'cd /d {repo}',
             f'{scratch / ("literal-" + mode + ".exe")} --gtest_output=xml:{repo / "docs/lfm/evidence" / ("literal-" + mode + ".xml")} > "{repo / "docs/lfm/evidence" / ("literal-" + mode + ".log")}" 2>&1', 'exit /b %errorlevel%']
    (scratch / ('run-' + mode + '.cmd')).write_text('\n'.join(lines) + '\n')
print(scratch)
