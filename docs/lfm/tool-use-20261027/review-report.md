# Independent LFM refit source review

Review scope: mutable candidate against a13dff47929bd40e38ecbb654ce7014e9a635704,
2026-10-27. Source review only; builds, tests and inference NOT_RUN by reviewer.
Other workers were changing disjoint files during review. Findings were sent to root
immediately. This report is evidence of review, not runtime acceptance.

## Important: terminal notification is delivered before the router's remainder

`src/llm/io_processing/output_parser.cpp:198-201` sends the request's finishReason
with only the first owned envelope, retaining the rest in its own buffer. New
`lfm2_tool_parser.cpp:324-325,397-405` makes that notification terminal and rejects
later nonempty chunks. Thus a single terminal chunk containing two native envelopes
can publish the first then throw while the owner drains the second.

Minimal integrated reproducer: construct OutputParser with registry `{g: object}`;
call parseChunk(`<|tool_call_start|>[g()]<|tool_call_end|><|tool_call_start|>[g()]<|tool_call_end|>`,
{}, true, STOP), then drain OutputParser until empty. Compare with the identical
text delivered as NONE chunks followed by empty STOP. Both should reconstruct two
distinct calls when parallel=true. Test STOP and LENGTH with complete second payload,
and missing second closer must publish no second call. Confidence HIGH by source flow;
reproduction NOT_RUN. Root was notified to add the integrated regression and fix the
terminal/remainder ownership without weakening truncation handling.

## Important: grammar and parser tool-name alphabets differed (fixed during review)

Initial `tool_grammar.hpp:54-61` accepted digits first and `-`/`.`; strict
`lfm2_tool_parser.cpp:275-283` accepts only native identifiers. A registered
`read-file`, `functions.cwd` or `1bad` could compile constrained output that the
retainer rejects. Root was notified. Current source now uses `return identifier(text)`;
new `LfmGenerationPolicy` tests include all three unsupported names. Verification of
the fix is root's pending test gate; reviewer ran no tests.

## Schema-profile alignment (fixed during review; tests pending)

Initial hard grammar accepted `const`, `examples` and `$comment`, while publication
uses the restricted Draft4 helper that rejects unsupported keywords. Current
`tool_grammar.hpp:192` now calls `isSupportedQwenXmlToolSchema` before generation.
This makes hard schema support an explicit intersection rather than allowing a
constrained call that always fails publication. New const/profile regression present.
The broader restricted Draft4 auto publication profile remains distinct from the
hard generation subset and must be described that way.

## Reviewed without additional source findings

- Complete native list retention is transactional before public IDs/arguments.
- Exact literal output is not serialized from the separate schema-validation DOM.
- `parallel=false` rejects multiple calls/envelopes; reset clears retained state.
- Canonical template override unwraps tools, preserves content before calls and
  reasoning aliases, while retaining the documented Minja string-argument seam.
- No forced lazy tool trigger is installed inside reasoning for auto.
- LFM absent-tools hard mode is rejected before request fallback to none.

## Remaining validation required

Full linked suites, generic router fragmentation/terminal regression, schema mask
runtime generation, live API string-argument replay, unary/SSE identity/finish reasons,
and repeated actual host-tool loops. The template's Minja compatibility branch emits
raw JSON argument strings as positional text if adaptation is bypassed; live candidate
evidence must establish that the OVMS probe/adapter takes the native mapping path.
