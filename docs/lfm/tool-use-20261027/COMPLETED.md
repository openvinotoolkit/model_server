# Completed reviewed deliverables

- A0–A5: locks, dossiers, retainer, templates, generation policy (see
  parser-report.md, template-report.md, policy-report.md).
- A6: candidate-final build COMPLETE; ovms_test filter LFM/Lfm/Qwen3Coder/Gemma4
  → 319 PASS / 1 SKIP (final-tests.xml).
- A7/A8 live API: protocol+policy-api PASS both GPU exports.
- A7/A8 agentic: native + action-required host-tool loops PASS 6/6 on
  LFM2.5-8B and LFM2-24B (runtime-native-required2-*). Control 8137 restored.

Truthful residual (not acceptance blockers for required agent profile):
deterministic+auto prose simulation; 24B native+auto parallel multi-call
fail-closed under parallel=false.
