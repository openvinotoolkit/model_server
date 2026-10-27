# LFM tool-use refit, 2026-10-27

ACTIVE_STAGE: A9 in progress — commit + residual root-cause delegate
BASE_SHA: a13dff47929bd40e38ecbb654ce7014e9a635704
CURRENT_HEAD: a13dff47929bd40e38ecbb654ce7014e9a635704 (dirty → A9 commit)
LAST_GREEN_TESTS: final candidate ovms_test 319 PASS / 1 SKIP; runtime-native-required2
PASS both models (native+action-required agent 6/6)
CURRENT_FAILURE: residuals only (deterministic+auto; 24B native+auto parallel)
NEXT_ACTION: land A9 commit/push; review residual-rootcause REPORT when ready
REVERT_POINT: BASELINE-LOCK.json; control 8137 READY
EVIDENCE_PATH: C:/gm/lfm-2026.4/tool-use-20261027
CANDIDATE: candidate-final/build-py_off ovms.exe sha256
95FB62B5DF7D68A880457378A4D3503CC606B5F06A82C23E70010D96358272BA
DELEGATE: residual-rootcause via cursor-agent model claude-opus-5-thinking-max
(read-only); hub under EVIDENCE_PATH/residual-rootcause/

Existing task worktree is reused; untracked scripts/windows/configs is pre-existing and must be preserved.
No changes to Gemma/Qwen implementations, model downloads, SDK/DLLs, RC3B/CIB2 or Pack.
Keep control server intact during source/build work. GPU validation runs models serially:
temporarily stop only exact locked control PID/command, use candidate loopback 18147,
then restore the saved control launch and verify health on 8137. This avoids GPU coexistence.
User explicitly authorized sustained implementation, external harness delegation and web search.
Execution uses the existing MFP-1 protocol, TDD and a review of each deliverable; no further design questionnaire.

## Design and acceptance

Retain native Python literal tool syntax; publish only complete registered schema-valid calls,
with exact numeric spelling and protocol-defined string escape semantics. Do not manufacture calls
from prose, reasoning, truncation or invalid candidates. Preserve unary/SSE parity and reset isolation.
Replay tool history through explicit model-specific overrides with real-tokenizer golden tests.
Introduce LFM-specific generation policy with explicit none/auto/required/named/parallel semantics;
unsupported hard modes fail closed. Auto must retain a no-tool path.
Reject generic parser rewrite and forced-auto-tool heuristic: both exceed native protocol semantics.
Parser, template, policy, build, live API and agentic behavior have separate verdicts.

## Plan

- [x] A0–A8 as previously recorded; runtime-native-required2 green
- [ ] A9: commits, FINAL-REPORT.md, residual root-cause delegate
