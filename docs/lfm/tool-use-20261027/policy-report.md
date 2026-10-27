# LFM generation policy

Status: source implemented; standalone policy tests PASS 12/12; whole server build and
live model masks NOT_RUN by this worker (root owns those stages).

Baseline defect: generation_config_builder.hpp has no lfm2 selector and silently selects
BaseGenerationConfigBuilder. Thus hard required/named modes install no native LFM grammar,
and the server's enable_tool_guided_generation flag is ineffective for LFM.

New tests: src/test/llm/output_parsers/lfm_generation_config_builder_test.cpp,
filter LfmGenerationPolicy.* (existing output-parser source glob).

Design: native Python-call EBNF for hard modes; no optional reasoning preamble, since the
official template leaves assistant content open, not an open thought. Auto retains the native
unconstrained path even when guidance is enabled. No unsafe trigger in an unclosed thought.
Schema profile and unsupported constraints fail explicitly. Fixed field order is a valid output
subset, not arbitrary-order generation support. Publication validation remains independent.

## Implementation

- generation_config_builder.hpp selects Lfm2GenerationConfigBuilder for lfm2.
- New lfm2/generation_config_builder.hpp installs hard grammar regardless of the optional
  guidance flag, marks requiresValidStructuredOutput true, rejects missing tools/named tools
  and response_format conflicts. Reusing a builder clears its previous hard grammar.
- New lfm2/tool_grammar.hpp emits one native envelope containing one or more registered
  Python-style calls. parallel=false has exactly one call; named selects only the chosen tool.
- Arguments use JSON double-quoted strings and JSON scalar/container values, a native literal
  subset accepted by LFM's literal reader. Optional fields have omission branches without
  dangling commas. Objects generate a canonical property-order subset and no unknown keys.
- Hard tool and parameter names must be ASCII identifiers, matching the native retainer.
  Dotted/hyphenated names and leading digits fail explicitly rather than generate rejected calls.
- Every selected schema must first pass the existing unchanged restricted Draft 4 executable
  validator isSupportedQwenXmlToolSchema. Grammar compilation then accepts a narrower profile:
  typed string/integer/number/boolean/null, objects/properties/required/boolean additionalProperties,
  arrays with one items schema, scalar enum, descriptions/title/default annotations and type unions.
  const is rejected by the executable profile. References, combinators, patterns, min/max and
  other unsupported keywords fail explicitly. Defaults are never used to supply required values.
- Bounds: 128 tools, 256-byte identifiers, 1 MiB per schema, schema depth 32, 2048 value nodes,
  128 fields per object, 8192 generated auxiliary rules, 256 scalar enum values; whitespace max 2.
- Auto is deliberately unconstrained and guidance=true does not force tool use. No lazy trigger
  is installed because reasoning spans can contain draft tool syntax. This is not a claim that
  autonomous model tool selection is repaired. Native raw auto behavior still needs live evaluation.
- Auto validates each active tool schema against the publication profile before generation;
  unsupported const/new dialect schemas fail at the request policy boundary. None skips that
  registry check and retains response_format. Auto remains unconstrained for supported schemas.
- Hard string escapes permit nonzero Unicode scalar escapes and paired high/low surrogates;
  lone surrogate escapes are excluded, matching LiteralReader. U+0000 escapes are excluded
  because the shared executable-schema validator rejects embedded NUL string values.

## Evidence

Commands (rtk wrapper required):

    rtk proxy python C:\gm\lfm-2026.4\tool-use-20261027\run-policy-tests.py red
    rtk proxy python C:\gm\lfm-2026.4\tool-use-20261027\run-policy-tests.py reset-red
    rtk proxy python C:\gm\lfm-2026.4\tool-use-20261027\run-policy-tests.py seams-red
    rtk proxy python C:\gm\lfm-2026.4\tool-use-20261027\run-policy-tests.py auto-profile-red
    rtk proxy python C:\gm\lfm-2026.4\tool-use-20261027\run-policy-tests.py green

Standalone MSVC uses the existing base generation-config library, pinned SDK include/import
libraries and RC4 runtime DLL directory, plus the current header implementation and new gtest file.
The runner stores exact compiler commands in policy-<stage>.cmd and libs in policy-libs.rsp.
All evidence lives under C:\gm\lfm-2026.4\tool-use-20261027.

- policy-red.log/xml: baseline selector, 7 tests, 5 FAIL / 2 PASS.
- policy-reset-red.log/xml: 9 tests, reuse leakage reproduced (1 FAIL).
- policy-seams-red.log/xml: 11 tests, profile/name mismatch reproduced (2 FAIL).
- policy-auto-profile-red.log/xml: 12 tests, unsupported auto profile reproduced (1 FAIL).
- policy-green.log/xml: 12/12 PASS. HardGrammarCompilesWithBothRealTokenizers constructs
  both target tokenizer fixtures (LFM2-2.6B is the existing 24B tokenizer alias) and invokes
  actual pinned GenAI validateStructuredOutputConfig. This proves grammar compilation, not
  token masking or tool completion by the model.

Build integration: generation_config_builders must export both new headers and depend on
io_processing_qwen3coder_tool_parser for the unchanged executable-profile validator.

## Remaining acceptance

Root must build candidate server/test binaries, run complete suites and actual hard/auto/none/named
unary/SSE model tests. The parser-retainer must reject nonregistered/incomplete/schema-invalid
calls independently. Agent execution and task completion remain separate from grammar correctness.
