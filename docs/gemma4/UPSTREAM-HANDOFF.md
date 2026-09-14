# Gemma4 2026.4 stack transfer — draft

Source: downstream RC `170644006a5334cb971b05824e4a8c95b495c4e2`. Target: upstream releases/2026/4 at `869b2186a004c6d7eba654db1b03b701bd80757f`.

## Scope

This transfers the custom registry-aware native/JSON tool parser, quoted call boundaries and nested arguments, independent reasoning parser, reasoning-to-tool routing, Google-template/history and rendered-prompt adaptation, auto/required/named guided grammars, parallel_tool_calls validation, opt-in session journal/seed state, and actual terminal streamer finish reasons/incomplete-frame diagnostics. Tool schemas carry max_whitespace_cnt=2 through the typed GenAI API.

Session persistence activates through OVMS_SESSION_STORE_DIR plus X-OVMS-Session-ID. Bodies are journaled on disk with bounded size/cache limits. This storage/API addition and endpoint-wide hard-choice validation need separate upstream design/security review and may need separate PRs.

The upstream logprob fix and all existing upstream Gemma4 parser tests are retained. Windows build-policy changes, other modalities, fork branding and later cache diagnostics/cache-off experiments are excluded. Generic utility helpers remain available for upstream users.

## Dependency blocker

The target GenAI dependency lacks JSONSchema(schema, optional whitespace_bound). The frozen GenAI patch is attached under dependencies for review; it is NOT applied by the upstream build. A companion GenAI API/serialization/matcher change and dependency update are required before this draft can compile. An unbounded fallback would invalidate the repair.

Downstream runtime tuple: OpenVINO `227c33757d1ef95d4da506d00686f923fdd2a535`, GenAI base `7ea2546852a382cd16bd22dea0cfad2db70ed744` plus attached patch, Tokenizers `a04accf6282d9b304214b492694b18c3979f667a`, XGrammar `9aa840b6d16abf094f3e8e2ac9c10465b77656c9`.

[Frozen package and provenance](https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS/releases/tag/gemmamonster-2026.4-rc-whitespace-17064400).

## Downstream live evidence

2026-09-14 22:44–22:49 UTC; already-running Windows RC, Intel Arc 140V 16GB, driver 32.0.101.8991, GPU/VLM_CB, Gemma4 26B A4B heretic INT4, prefix caching and DEBUG enabled. Concurrent client traffic was not controlled.

| Check on frozen RC | Result |
|---|---|
| Non-stream benchmark | 14/14 HTTP 200 |
| Long requests, curl | 1797 actual tokens / 72.979 s = 24.624 tok/s |
| Long requests, Invoke-WebRequest | 2186 actual tokens / 86.639 s = 25.231 tok/s |
| Named single echo | PASS, exact arguments, 1 call, tool_calls finish |
| Two same-name parallel echo calls | PASS, exact arguments, 2 calls, tool_calls finish |
| Named SSE echo | PASS, reconstructed arguments, 1 call, tool_calls finish |

Raw synthetic requests/responses and summaries are under evidence. Tool cases: temperature=0, seed=170644, max_tokens=256. Benchmark: temperature=1, top_k=64, top_p=0.95, preserved seeds/prompts; timing includes prefill/HTTP/possible queueing. Long outputs stopped before max_tokens and metrics use actual usage. Cold startup, TTFT, concurrency, factual accuracy of free benchmark texts, multi-turn/real-tool execution and session persistence were NOT RUN in this campaign.

These results belong to the frozen downstream RC, NOT the assembled upstream head. The original candidate manifest's historical live acceptance NOT_RUN is not overwritten.

## Merge gates

- Review/land the companion GenAI API and update dependencies coherently.
- Build product and run the six targets below on this exact PR head.
- Run every retained upstream parser regression and generic parser/streamer coverage plus Linux/Windows CI; no unsupported exclusions.
- Review journal filesystem/seed/API behavior and hard-choice policy; split scope if maintainers prefer.
- Run packaged repeated, streaming, multi-turn/tool-result semantic acceptance with raw evidence on this head.

```text
//src/test/llm/gemma4_fast:gemma4_parser_contract_test
//src/test/llm/generation_config:gemma4_generation_contract_test
//src/test/llm/generation_config:gemma4_prompt_state_generation_contract_test
//src/test/llm/generation_config:openai_parallel_tool_calls_contract_test
//src/test/llm/gemma4_overlay:gemma4_chat_template_overlay_contract_test
//src/test/llm/gemma4_overlay:gemma4_google_jinja_contract_test
```

Build, executable tests and broad CI on the transferred head: NOT RUN. This remains draft while gates are open.
