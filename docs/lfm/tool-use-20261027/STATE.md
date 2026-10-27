# LFM tool-use refit, 2026-10-27

ACTIVE_STAGE: A9 DONE; Design C APPROVED (docs); memory-pressure/u4 matrix DONE; **Hybrid C C++ NOT started**
BASE_SHA: a13dff47929bd40e38ecbb654ce7014e9a635704
CURRENT_HEAD: see branch tip after docs/config handoff commit
LAST_GREEN_TESTS: final-tests 319/1 SKIP; runtime-native-required2 PASS both models
CURRENT_FAILURE: residuals + Design C leak + offload cliff (see HANDOFF-CLOUD-REVIEW.md)
STAFF_MODE: parallel_tool_calls=true (API default already; agent/probe harnesses aligned)
NEXT_ACTION: cloud review of Design C + speed/u4; then implement Hybrid C plan tasks 1–4; optional start-lfm25 u4 default
REVERT_POINT: BASELINE-LOCK.json
EVIDENCE_PATH: C:/gm/lfm-2026.4/tool-use-20261027
CLOUD_HANDOFF: docs/lfm/tool-use-20261027/HANDOFF-CLOUD-REVIEW.md
PR: https://github.com/openvinotoolkit/model_server/pull/4697

## Design C (Hybrid) — thought / tool separation

Approved 2026-10-27. Keep `<think>` primary; demote pseudo-plan JSON / `[tool:…]` prose into `reasoning_content`; execute only native envelopes.

- Spec: `docs/superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md`
- Plan: `docs/superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md`
- Implementation: **NOT STARTED**

## Speed / memory pressure (LFM2.5)

- Cliff: r0≈55 → r20≈31.5 → r30≈24 tok/s (`speed-offload-20261027/SUMMARY-v4.json`)
- u4 matrix DONE: r0+u4≈58.5 (+3.5); r20+u4≈32.4 (cliff remains ~−23 vs r0)
- Recs: default r0+u4; r20+u4 = VRAM trade not speed fix; offload needs WEIGHTS_PATH+CACHE_DIR
- Report: `C:/gm/lfm-2026.4/tool-use-20261027/speed-offload-20261027/MEMORY-PRESSURE-REPORT.md`

## Plan
- [x] A0–A9 including commit/push and residual root-cause (read-only)
- [x] Design C spec + plan written / approved
- [x] Memory-pressure / u4 matrix + report
- [x] Cloud handoff docs for PR 4697
- [ ] Hybrid C implementation (see plan tasks 1–4)
- [ ] Optional: wire KV_CACHE_PRECISION=u4 into start-lfm25.ps1
