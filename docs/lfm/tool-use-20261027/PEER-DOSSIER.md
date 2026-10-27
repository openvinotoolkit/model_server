# MFP-1 A1/A2: LFM tool protocol dossier

Captured 2026-10-27. Exact URLs, resolved revisions and SHA256 hashes are in
`C:/gm/lfm-2026.4/tool-use-20261027/upstream/SOURCES.lock.json`.
These are source comparisons, not peer-server differential runtime acceptance.

| Source / mechanism | Decision | Target evidence |
| --- | --- | --- |
| LiquidAI LFM2.5-8B-A1B `5dd22602c2e9f6a097b1de4c4efe0658b605015c`: plain JSON list of function definitions, native Python-call list, tool role replay, reasoning aliases | PORT SEMANTICS | explicit per-model overrides; `LfmAgentTemplate*` real-tokenizer/Minja tests |
| LiquidAI LFM2-24B-A2B `a3bbacd91a678b97712f0e323e52f8c24ba29542`: tool-list/response markers, content before calls, null-safe content, native history | PORT SEMANTICS | same tests with distinct target tokenizer and marker golden |
| vLLM `fe77d0906be0c38c5e2c575cd61b545a2fa687e4`: keep special tool delimiters visible | KEEP semantics | LFM token-aware router; quoted marker split tests |
| vLLM: `make_valid_python`, quote/control/leading-zero repair before AST conversion | DROP | retainer rejects malformed closed envelopes and never fabricates missing values/closers |
| vLLM: suppress model echo of a previous call body in trailing prose | UNKNOWN; no heuristic port | trailing visible content is preserved; no content deduplication claim |
| SGLang `c43abd6d9e38a14e81bdb77c648a06353e1c10ee`: buffer complete native envelopes before extraction | KEEP semantics | transactional list validation, no speculative identity |
| SGLang: alternate JSON envelopes, renamed Python-reserved parameters, leading-zero and quote repair | TOLERANCE ONLY / NOT PORTED | canonical native keyword literals only; unsupported hard names/schemas fail explicitly |
| llama.cpp `7aca0d2ea8d92b56911233b822db01f9fbb1ae50`: distinguish LFM2 special list tokens from LFM2.5 plain list | KEEP semantics | separate model overrides and exact token golden assertions |
| Official template raw JSON-string arguments vs pinned Minja capability detection | REFINE | retain detection compatibility branch; OVMS adapter maps API JSON arguments to native keyword history; live replay gate pending |

## Channel and policy contract

Native envelope: `<|tool_call_start|>[name(key=value)]<|tool_call_end|>`.
LFM2.5 reasoning uses `<think>` / `</think>`; tool-looking text inside open reasoning
stays reasoning. No Gemma-style tool handoff is introduced. The canonical assistant
generation suffix is `<|im_start|>assistant\n`; no canonical thinking-off suffix is claimed.

`none` retains existing response_format handling and suppresses executable tools at the API.
`auto` remains a native unconstrained reference, including a no-tool path. It is not
an obligation to call a tool. `required` and named use a root native EBNF envelope from
the first constrained token; their grammar compilation failures must not fall back to auto.
Hard mode intentionally skips optional model reasoning. `parallel=false` has both grammar
and retainer checks. Named API filtering supplies only the selected registered schema.

Hard grammar support is the intersection of the unchanged restricted Draft4 publication
validator and the narrower LFM grammar subset. Numeric bounds, regex, references and other
unimplemented grammar constraints are rejected explicitly in hard modes. This is not full
JSON Schema support. Auto publication retains the validator's broader restricted Draft4
profile and validates emitted values without serializing the validation DOM over raw JSON.

## Known incident regression and acceptance boundaries

The supplied user transcript is a behavioral failure: instructions instead of execution,
one successful write, no verified environment/program launch. Our baseline additionally
reproduces hard-policy API failures with saved requests/responses. A retainer cannot make
a model choose an appropriate action; actual host-tool loops remain their own acceptance gate.

Adversarial peer-inspired controls include raw control/escape repairs, leading-zero numbers,
quoted end markers and incomplete calls. Unsupported alternate envelope tolerance is not
silently accepted. Peer server execution and cross-runtime differential runs: NOT_RUN.
