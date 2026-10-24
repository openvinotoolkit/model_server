# Gemmamonster current baseline

The baseline for new work is `gemmamonster-2026.4-rc4`, published in the releases of `DassaultFalconKing/gemmamonster_model_server_OVMS`, with the same candidate source integrated into fork main. See [RC4 Windows release](qwenmonster/RC4-WINDOWS-RELEASE.md) for composition, packaging, manifests and acceptance limits.

Both Windows variants are real builds: `win_mp_on_py_on` and `win_mp_on_py_off`. The selected runtime retains the protected custom GPU plugin `21a2fd9f…` with the 2026.4 RC1 SDK cohort. The original measured EXE is archived separately; its performance numbers are not assigned to a rebuilt EXE.

Qwen3Coder implements literal-string and numeric/container spelling retention derived from the separate Gemma4 reference. “Anti-hallucination parser-retainer” means preserving emitted payloads and rejecting invalid/incomplete structure without inventing arguments; it does not guarantee a model's factual correctness. Gemma parser code and its acceptance are not reopened in this release preparation.

Previous main history remains reachable through the RC4 integration parent and `archive/main-before-rc4-20261024`. Older release/parser branches and local build directories are recovery/evidence sources, not active baselines. Active CIB2, OTD, MiniCPM and the frozen C6/2026.5 line are separate work; they are not silently merged or deleted. New changes must explicitly port onto RC4 and establish fresh evidence.

This index supersedes historical current-state declarations. It does not upgrade focused package/model checks into full regression, agentic, long-context or soak acceptance.
