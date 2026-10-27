# Cloud handoff — LFM tool-use + Design C + speed/memory

**PR:** https://github.com/openvinotoolkit/model_server/pull/4697  
**Branch:** `dev/lfm-family-hardening-20261026`  
**Repo:** `C:\git\gemmamonster-lfm-2026.4`  
**Date:** 2026-10-27  

Ask cloud reviewers to: (1) review **Design C** (Hybrid thought demotion) before product C++ starts; (2) acknowledge the **offload cliff + u4** speed findings; (3) keep residual tool-use UX items on the radar.

---

## Summary

A9 tool-use retainer / generation policy / templates landed and were accepted under **native + action-required** on LFM2.5-8B and LFM2-24B. Staff mode uses `parallel_tool_calls=true`. Free-form “thought” / pseudo-plan JSON still leaks into assistant `content` under unconstrained `auto` — **Design C is APPROVED, implementation NOT started**. Offload decode cliff is measured; **u4 helps r0**, does **not** close r20.

| Area | Status |
|------|--------|
| Retainer + policy + templates (A0–A9) | DONE (in PR) |
| Design C Hybrid thought demotion | **APPROVED / NOT STARTED** |
| Speed offload cliff (SUMMARY-v4) | MEASURED |
| Memory-pressure / u4 matrix | **DONE** (report outside git) |
| Peer-server differential | NOT_RUN |

---

## Design / plan links (in this PR)

- Spec: [`docs/superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md`](../../superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md)
- Plan: [`docs/superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md`](../../superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md)
- Prior in-repo handoff: [`HANDOFF-REVIEW.md`](HANDOFF-REVIEW.md)
- State: [`STATE.md`](STATE.md) · Final A9: [`FINAL-REPORT.md`](FINAL-REPORT.md)

**Design C (approved):** keep `<think>` primary; demote closed pseudo-plan JSON (`explanation` / `plan` / `commands` …) and `[tool:…]` prose into `reasoning_content`; execute **only** native `<|tool_call_start|>…<|tool_call_end|>`. No forced-auto. Demotion ON only when both parsers are `lfm2`.

---

## ALL open problems (complete set)

### 1. Design C Hybrid thought demotion — APPROVED, **implementation NOT started**

Product C++ (`thought_demotion.hpp`, retainer buffer flush, tests) has **not** begun. Spec + plan are in-tree for cloud review before coding.

### 2. LFM2.5 free-form “thought” / pseudo-plan JSON leaks into `content`

Under `tool_choice=auto`, model emits non-Liquid plan blobs (`explanation` / `plan` / `commands`, `/workspace` bash) and `[tool:…]` prose. Retainer correctly refuses to execute them; they still appear as visible `content`. Needs Design C. Evidence: residual-rootcause + deterministic+auto agent FAIL runs under `C:\gm\lfm-2026.4\tool-use-20261027\`.

### 3. Speed cliff under offload (LFM2.5)

| Config | Decode median | vs r0 (~55) |
|--------|---------------|-------------|
| r0 (task5 baseline) | ~55 | 0 |
| r20 + WEIGHTS + CACHE (SUMMARY-v4) | **31.53** | **−23.5** |
| r30 + WEIGHTS + CACHE (SUMMARY-v4) | **23.96** | **−31.0** |

Evidence: `C:\gm\lfm-2026.4\tool-use-20261027\speed-offload-20261027\SUMMARY-v4.json`  
Offload **requires** `WEIGHTS_PATH=…/openvino_model.bin` + `CACHE_DIR` (not `openvino_language_model.bin`).

### 4. Memory-pressure / u4 matrix — **DONE** (does not close cliff)

Authoritative: `...\speed-offload-20261027\MEMORY-PRESSURE-REPORT.md` + `MEMORY-PRESSURE-SUMMARY.json`.

| Config | Decode med | vs r0 (~55) |
|--------|------------|-------------|
| **r0 + u4** | **58.54** | **+3.5** |
| r20 + u4 | 32.41 | −22.6 (only **+0.9** vs plain r20) |
| r20 + u4 warm CACHE_DIR | 31.93 | cliff remains |
| r20 + u4 long profile | 31.49 | no help |
| r20 + u4 seqs2 | 29.53 | **worse** |
| r20 + u4 dyn KV | 31.33 | no help |

**Recommendations for start-lfm25 (not yet applied in script pin):**

- Default: always `KV_CACHE_PRECISION=u4` at r0 (speed win).
- Treat **r20+u4 as VRAM trade, NOT a speed fix** for the ~−23 tok/s offload cliff.
- Do not expect warm CACHE_DIR / long-profile / seqs2 / dyn-KV to recover decode on this short bench.
- When offload > 0: still require `WEIGHTS_PATH=openvino_model.bin` + stable `CACHE_DIR`.

### 5. `tool_choice=none` final prose hang after successful write + `run_python`

Code-demo agent completes tools then would hang on a final prose turn with `tool_choice=none`. Harness **workaround only**: `skip final prose (tools done)`. Not a retainer bug; product UX / generation behavior still open. Evidence: `C:\gm\lfm-2026.4\tool-use-20261027\code-demo-runner4.log`, `code-demo-lfm25\agent\`.

### 6. Residuals from A9 / PENDING / residual-rootcause

- **deterministic+auto:** prose simulates tools; unconstrained auto by design (HIGH confidence model/harness, not parser).
- **24B native+auto under `parallel=false`:** historical fail-closed multi-call (`Parallel LFM tool calls are disabled`). Staff default is now `parallel_tool_calls=true`; `false` remains explicit opt-in fail-closed.
- Report: `C:\gm\lfm-2026.4\tool-use-20261027\residual-rootcause\REPORT.md` (NEEDS_REVIEW → accepted as analysis).
- In-repo: `PENDING.md`, `FINAL-REPORT.md`, `COMPLETED.md`.

### 7. Staff mode `parallel_tool_calls=true`; `parallel=false` still fail-closed by design

API default + agent/probe harnesses aligned to `true`. Retainer still rejects multi-call when client sets `false`. Do not weaken.

### 8. `plugin_config` JSON escaping pitfalls on Windows `ProcessStartInfo`

PS 5.1 / Legacy native arg passing needs `\"` escaping or call-operator + `ConvertTo-Json -Compress`. Broken escaping caused prior offload load failures (see speed-offload v3 LOADING_PRECONDITION_FAILED). Documented in MEMORY-PRESSURE-REPORT recommendations.

### 9. Peer-server differential

`PEER-DOSSIER.md`: **NOT_RUN**. Do not treat as done.

---

## Evidence paths (outside git — summarized here, not committed)

Root: `C:\gm\lfm-2026.4\tool-use-20261027\`

| Artifact | Path |
|----------|------|
| Final unit suites | `final-tests.xml` (319 PASS / 1 SKIP) |
| GPU native+required | `runtime-native-required2-lfm25`, `runtime-native-required2-lfm2-24b` |
| Residual root-cause | `residual-rootcause\REPORT.md` |
| Offload cliff | `speed-offload-20261027\SUMMARY-v4.json` |
| Memory / u4 | `speed-offload-20261027\MEMORY-PRESSURE-REPORT.md`, `MEMORY-PRESSURE-SUMMARY.json` |
| Per-cell mem runs | `speed-offload-20261027\mem-r0-u4\`, `mem-r20-u4*\` |
| Code-demo hang skip | `code-demo-runner4.log`, `code-demo-lfm25\agent\` |
| Candidate EXE | `candidate-final\build-py_off` (sha256 `95FB62B5…7272BA`) |

---

## Config / launch pins in this commit

- `scripts/windows/start-lfm25.ps1` — pins **candidate-final** ovms.exe + hash (still `OFFLOAD_RATIO=0`; **u4 default not yet wired** — follow MEMORY-PRESSURE recommendations).
- `scripts/windows/start-lfm2-24b.ps1` — already on branch (long profile).
- `scripts/windows/configs/lfm2-24b/command.json` — control command snapshot (OFFLOAD 0).

---

## Suggested cloud review checklist

- [ ] Design C demotion profile closed enough? (keys, leading-prose ≤200, no promotion to `tool_calls`)
- [ ] Confirm demotion stays OFF for LFM2-24B tool-only / when reasoning parser ≠ `lfm2`
- [ ] Accept offload cliff as residency trade; prefer r0+u4 when VRAM allows
- [ ] Do **not** “fix” auto agent FAIL by parsing prose into tools
- [ ] Keep `parallel=false` fail-closed; staff `true` is intentional
- [ ] Next product work: Hybrid C plan tasks 1–4 (TDD), then optional start-lfm25 u4 param sketch

---

## Non-goals for this handoff

- No true-orphan evidence branch (product-adjacent docs/config go on PR 4697).
- No Pack/CIB2 engine changes, no Gemma/Qwen rewrites, no control-8137 surgery in Design C.
- External `C:\gm\…` evidence stays outside git.
