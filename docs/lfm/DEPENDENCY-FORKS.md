# Dependency fork review

User steering: consider our minja and GenAI forks as possible solutions; do not assume
that a fork helps. Read-only inspection on 2026-10-26; no SDK or binary replacement.

| Component | Verified source | Relevant behavior | Current decision |
| --- | --- | --- | --- |
| minja | C:\gm\qwenmonster\minja, 83eb074e42140b0d1194d1dd61f29c31e4515ee9; public codex/c6-head-20261019-minja-locked matches via git ls-remote | c71bdef fixes adjacent Jinja string concatenation; 83eb074 changes native Gemma 4 capability probing | Neither fixes an LFM macro that concatenates unescaped string data; LFM override minja tests already pass on the pinned runtime |
| minja later template fixture | Public 1bdb178afd6f527354e17a1db1b3a22b59189f21 | Gemma 4 JSON-schema union rendering in a test template | No LFM fix identified; do not substitute this cohort |
| GenAI 2026.4 fork | Public fix/schema-whitespace-bound-2026.4, 33bfec6e44d5c2a1bc5ad12c375f65d59929da85; object inspected locally in C:\gm\c6\genai and public ref verified | Optional JSONSchema max_whitespace_cnt plus XGrammar pin | Candidate only if actual guided-generation failures need it; not a string-literal parser or template escaping repair |
| GenAI later local fork | C:\gm\c6\genai HEAD 294f698d; 9a5efc0f token bridge, f82bd6a6 bindings, 294f698d JavaScript factories | Token-aware structured tags / formats | Source lead for a demonstrated guidance blocker; not a 2026.4 binary input and not transplanted |

The 2026.4 companion explicitly changes the C++ JSONSchema struct layout and requires
a coordinated consumer rebuild; it is not an ABI-compatible header/DLL swap.
See [exact source patch](https://github.com/DassaultFalconKing/openvino.genai/commit/33bfec6e44d5c2a1bc5ad12c375f65d59929da85).

The existing OVMS factory has no LFM-specific GenerationConfigBuilder: setting
enable_tool_guided_generation for lfm2 selects the base builder. A model obeying a
named tool request is therefore not evidence of enforced LFM grammar constraints.
Keep actual tool-cycle success separate from guided-generation capability. Introduce
no dependency changes without a corresponding observed need and paired 2026.4 tests.

Runtime still uses the RC4-locked GenAI 2026.4.0.0-3401-5f7f1278107 DLL and pinned
Core/Tokenizers. Exact hashes and versions are in evidence/PREFLIGHT.json and
evidence/runtime-versions.json. C6, RC3B and CIB2 sources/binaries were not modified.
