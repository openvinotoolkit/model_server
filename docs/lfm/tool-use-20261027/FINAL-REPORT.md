# LFM tool-use refit — final report (A9)

Date: 2026-10-27. Base: `a13dff47929bd40e38ecbb654ce7014e9a635704`.
Candidate: `C:\gm\lfm-2026.4\tool-use-20261027\candidate-final\build-py_off`
(`ovms.exe` sha256 `95FB62B5DF7D68A880457378A4D3503CC606B5F06A82C23E70010D96358272BA`).

## Separate verdicts

| Gate | Verdict | Evidence |
|------|---------|----------|
| Parser retainer | PASS | standalone 16/16; integrated in final-tests |
| Templates / replay | PASS | template-report; LfmAgentTemplate in final-tests |
| Generation policy | PASS | standalone 12/12; live policy-api both models |
| Build / modules | PASS | candidate-final COMPLETE; final-tests.xml |
| Live API unary/SSE/tool/continuation | PASS | runtime-native-required2-* |
| Live policy matrix | PASS | policy-api RESULT both exports |
| Agentic host-tool loops | PASS under **native + action-required** | 6/6 × both models |
| deterministic+auto agent | FAIL (residual) | prose simulates tools; auto no-tool path by design |
| 24B native+auto agent | FAIL (residual) | fail-closed: Parallel LFM tool calls disabled |

## What landed in source (Codex workers + root integration)

See also `parser-report.md`, `policy-report.md`, `template-report.md`, `review-report.md`.

### Tool parser (retainer)

- Registry-taking `Lfm2ToolParser` publishes only after a complete native
  `<|tool_call_start|>[...]<|tool_call_end|>` envelope; transactional multi-call
  lists; schema/registry validation before public IDs.
- `parallel=false` rejects multi-call / second envelopes in one turn.
- Bounds 1 MiB / 4096 calls; no synthetic closers on STOP/LENGTH.
- Router fix: non-final segments get NONE so a terminal chunk can drain multiple
  envelopes without premature terminal lock.
- Fixtures migrated to explicit tool registries; streaming helpers compare totals.

### Generation policy

- `lfm2` selector → `Lfm2GenerationConfigBuilder` + `tool_grammar.hpp` EBNF.
- Hard required/named install native grammar even when guidance flag optional;
  auto stays unconstrained (no forced-auto heuristic).
- Hard names = ASCII identifiers; schema must pass
  `isSupportedQwenXmlToolSchema` then a narrower grammar profile.
- Live: none/auto/required/named/negative paths exercised via
  `lfm-tool-use-probe.py`.

### Templates / smoke / agent harness

- Model-specific overrides for tool-history replay; copied into extras/test trees.
- `lfm-smoke.py --tool-use-refit` runs policy + agent loops.
- Agent payload fixed to `O'Brien C:/llm/lfm-test 1.10` (avoid `\t` JSON/Python trap).

## Outstanding (truthful)

1. **deterministic+auto**: model invents `[tool:…]` prose instead of structured
   calls — expected when auto retains unconstrained generation.
2. **LFM2-24B native+auto under parallel=false** (historical residual): multi-call
   fail-closed. **Staff mode is now `parallel_tool_calls=true`** (API default and
   harness default); false remains an explicit client opt-in. Root-cause report:
   `C:\gm\lfm-2026.4\tool-use-20261027\residual-rootcause\REPORT.md`.

## Control server

GPU runs temporarily stop the locked control OVMS on 8137, validate on 18147,
restore exact BASELINE-LOCK command. Last restore: READY.
