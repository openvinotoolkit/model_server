# LFM retainer implementation and evidence

Base: a13dff47929bd40e38ecbb654ce7014e9a635704. Source changes are uncommitted for root integration.

## Reproduced baseline

Standalone MSVC/gtest binary linked the existing built LFM parser library. Both new tests
failed semantically: the legacy parser published identity at `[f(` and arguments at `[f(a=1)`
before native list/envelope completion. Evidence: `C:/gm/lfm-2026.4/tool-use-20261027/red.log`
and `red.xml`. Initial loader attempt had empty output because `serving` contains only the
EXE; corrected runtime PATH uses pinned RC4 extracted DLL directory. That empty run is not RED evidence.

## Implementation

Registry-taking `Lfm2ToolParser(tokenizer, schemas, parallelToolCalls=true, optional config)`
is the executable publication path. It snapshots schema text, retains a complete native
`<|tool_call_start|>[name(arguments),...]<|tool_call_end|>` envelope, parses all calls
transactionally, validates registration and arguments, and only then allocates distinct
public IDs and returns complete deltas. A malformed later call invalidates that entire list.
`parallel=false` rejects multiple calls and second envelopes within one turn.

Native literal decoding remains `lfm2::parseArguments`: protocol escapes are decoded once,
numeric spelling and nested numeric text remain exact. Validation reuses the existing
restricted Draft4 `validateQwenXmlToolArguments` helper without changing Qwen source.
Its separate validation DOM is never serialized over retained arguments. Unsupported
schema profiles fail closed; this is not a full arbitrary-dialect JSON Schema claim.
Root/policy worker must align hard-policy admission with `isSupportedQwenXmlToolSchema`.

Retention is limited to 1 MiB and 4096 calls per generation, across envelopes. The incremental quote-aware
closer scanner visits retained bytes once except bounded delimiter suffix checks.
STOP, LENGTH and TOOL_CALL never synthesize list/envelope closing boundaries. This SDK
has no cancellation/error finish enum; the owner resets cancelled/failed parsers.
`resetState` clears candidates, queue, terminal state, indexes and allocated ID set.
The no-registry constructor remains explicitly labeled low-level legacy compatibility;
production must select the registry overload, including an empty registry.

## Tests and integration

`src/test/llm/output_parsers/lfm_retainer_test.cpp`: 16 tests PASS in standalone current-source
build, including all two-way splits, byte chunks, quoted end markers, number/string fidelity,
unknown tools, schema types/required/additional properties/defaults, whole-list rejection,
zero arguments, repeated identities, parallel constraints, truncation, cancellation reset,
unsupported schema and buffer limits.

Command: `rtk proxy python C:/gm/lfm-2026.4/tool-use-20261027/run-retainer-tests.py green`.
Evidence: `green-compile.log`, `green.log`, `green.xml` in that directory. The runner compiles
the current LFM source against pinned SDK and existing schema/base/helper libraries, not a
fresh entire OVMS binary. Only the initial two invariant tests have recorded baseline RED;
additional cases were added during implementation and are not separately claimed RED.

Updated LFM2/LFM25/strict integration fixtures to explicitly register their tool names.
Existing streaming helpers compare reconstructed content/reasoning/tool totals rather than
requiring speculative per-chunk identities. Malformed closed envelopes remain covered and
expect explicit exceptions; incomplete envelopes remain non-executable.

Root owns: BUILD registration/dependency, production selector/parallel argument propagation,
fresh full build, focused/full integration suites, API/SSE tests and runtime evidence.
Those gates are NOT_RUN by this worker. No Gemma/Qwen implementation, SDK, deployed server,
downloaded models or protected build was changed.

Router seam found during review: a terminal raw chunk may contain several envelopes and content.
The generic router splits at the first closer but used to send STOP prematurely to that first
owned segment. Strict parser retains its terminal lock; root owns a router fix that forwards
NONE to non-final segments and preserves the owner terminal notification. Integrated regression
`TerminalChunkDrainsMultipleEnvelopesAndContent` was added for both target tokenizers and awaits
root's full-build GREEN. Lightweight integrated RED was reproduced for both target tokenizers:
`router-red.log`/`router-red.xml` show `LFM output received after terminal notification`.
Runner `run-retainer-integration.py` compiles current router/LFM sources against existing pinned libraries.
Do not attribute interim unlocked-parser runs to the final guarded source.

Global bound follow-up: `BoundsAllPublishedCallsAcrossEnvelopesUntilReset` feeds 2048 two-call
envelopes without draining, then requires rejection at call 4097 and successful reset. Unique
`bound-red.log`/`bound-red.xml`: 15 PASS, 1 FAIL because the prior per-envelope bound allowed
unbounded identities/pending deltas. Guard now counts published calls plus the entire pending
envelope before publication; `bound-green.log`/`bound-green.xml`: 16/16 PASS.
Source freeze: 2026-10-27 02:14:48 UTC; root must include this final source in incremental build.

Fixture follow-up: lightweight current-source integration initially ran 101 tests: 98 PASS,
1 SKIP, 2 FAIL, both `StreamingWithContentBetweenToolCalls`, because the extraction registry
omitted `solve`. Added that explicit tool name in both fixtures; no production fallback.
Final `fixtures-green.log`/`fixtures-green.xml`: 100 PASS, 1 SKIP, 0 FAIL, including both
terminal multi-envelope cases and both `StreamingWithToolCallAndFinishReason` cases with root's
router fix. The skip is the existing LFM2-only reasoning non-capability.
This lightweight build compiles current router/LFM/fixture sources and links pinned existing
base/helper/streamer/schema libraries. It is not fresh full-server runtime acceptance.
Harness recovery: the first combined fixture compile lacked `/Zc:preprocessor` and then lacked
`text_streamer.lib`/`openai_delta_serializer.lib`; these compile/link failures were corrected in
the standalone runner, not attributed to product source.
Final fixture source freeze: 2026-10-27 02:23:44 UTC.
