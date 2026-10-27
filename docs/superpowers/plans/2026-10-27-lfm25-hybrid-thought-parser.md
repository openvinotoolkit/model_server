# LFM2.5 Hybrid C Thought / Tool Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demote LFM2.5 pseudo-plan JSON and pseudo-tool prose into `reasoning_content` while keeping `<think>` as the primary thought path and publishing `tool_calls` only from native Liquid envelopes.

**Architecture:** Add a pure classifier in `lfm2/thought_demotion.hpp`. When `tool_parser=lfm2` and `reasoning_parser=lfm2`, the registry `Lfm2ToolParser` buffers non-envelope segments and emits `ReasoningDelta` or `ContentDelta` per the closed demotion profile. Retainer publication rules stay unchanged.

**Tech Stack:** C++17, gtest, existing OVMS OutputParser / Delta variant, LFM2.5 tokenizer fixtures under `src/test/llm_testing/LiquidAI/`.

**Spec:** `docs/superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md`

## Global Constraints

- Do not promote `[tool:…]`, plan `commands`, or any non-native syntax into `tool_calls`.
- Do not add forced-auto grammar or weaken the retainer.
- Demotion ON only when both parsers are `lfm2`; LFM2-24B tool-only remains demotion OFF.
- Do not edit Gemma/Qwen/Onyx parsers, Pack/offload, or stop control OVMS on port 8137.
- v1 holds demotion-enabled non-envelope text until segment close (no optimistic content flush).
- Demotion key set is fixed: `explanation`, `plan`, `commands`, `command`, `steps`.
- Leading prose before plan JSON ≤ 200 bytes (D1); otherwise keep as content unless D2 matches.
- Share the existing 1 MiB retainer byte limit; clear demotion buffer in `resetState()`.
- Prefer finishing unit/integration tests green before any live GPU re-smoke.

## File structure

| File | Responsibility |
| --- | --- |
| `src/llm/io_processing/lfm2/thought_demotion.hpp` | Pure classifier + helpers (`shouldDemote`, key scan). Header-only OK. |
| `src/llm/io_processing/lfm2/lfm2_tool_parser.hpp` | `demotePseudoThought` flag, demotion buffer, `flushDemotionBuffer`. |
| `src/llm/io_processing/lfm2/lfm2_tool_parser.cpp` | Buffer non-envelope text when demotion on; flush as Reasoning/Content delta. |
| `src/llm/io_processing/output_parser.cpp` | Pass `demotePseudoThought=(reasoningParserName=="lfm2")` into Lfm2ToolParser. |
| `src/llm/BUILD` | Add `thought_demotion.hpp` to `io_processing_lfm2_tool_parser` hdrs. |
| `src/test/llm/output_parsers/lfm_thought_demotion_test.cpp` | Pure classifier unit tests (no tokenizer). |
| `src/test/llm/output_parsers/lfm_strict_output_parser_test.cpp` | Integrated demotion + think + native envelope cases. |

---

### Task 1: Pure demotion classifier (TDD)

**Files:**
- Create: `src/llm/io_processing/lfm2/thought_demotion.hpp`
- Create: `src/test/llm/output_parsers/lfm_thought_demotion_test.cpp`
- Modify: `src/llm/BUILD` (`io_processing_lfm2_tool_parser` hdrs)

**Interfaces:**
- Consumes: nothing (standalone)
- Produces:
  - `namespace ovms::lfm2` with:
    - `inline constexpr size_t kDemotionLeadingProseLimit = 200;`
    - `bool shouldDemoteSegment(std::string_view segment);`
    - Helpers may be file-static or in the same namespace; keep `shouldDemoteSegment` as the only required public entry for later tasks.

- [ ] **Step 1: Write the failing classifier tests**

Create `src/test/llm/output_parsers/lfm_thought_demotion_test.cpp`:

```cpp
// Copyright 2026 Intel Corporation. SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "src/llm/io_processing/lfm2/thought_demotion.hpp"

TEST(LfmThoughtDemotion, OrdinaryProseKept) {
    EXPECT_FALSE(ovms::lfm2::shouldDemoteSegment("The capital of France is Paris."));
}

TEST(LfmThoughtDemotion, WellFormedPlanJsonDemoted) {
    const char* plan =
        R"({"explanation":"need sandbox","plan":["cwd","write"],"commands":["bash /workspace"]})";
    EXPECT_TRUE(ovms::lfm2::shouldDemoteSegment(plan));
}

TEST(LfmThoughtDemotion, MalformedPlanJsonStillDemoted) {
    const char* broken =
        R"({"explanation": "draft", "plan": [unterminated, "commands": ["bash /workspace"]})";
    EXPECT_TRUE(ovms::lfm2::shouldDemoteSegment(broken));
}

TEST(LfmThoughtDemotion, LeadingProseWithinLimitThenPlanDemoted) {
    std::string text(200, 'x');
    text += R"({"explanation":"x","plan":[]})";
    EXPECT_TRUE(ovms::lfm2::shouldDemoteSegment(text));
}

TEST(LfmThoughtDemotion, LeadingProseOverLimitKept) {
    std::string text(201, 'x');
    text += R"({"explanation":"x","plan":[]})";
    EXPECT_FALSE(ovms::lfm2::shouldDemoteSegment(text));
}

TEST(LfmThoughtDemotion, PseudoToolLinesDemoted) {
    EXPECT_TRUE(ovms::lfm2::shouldDemoteSegment("[tool:cwd] /home/user\n[tool:write] path"));
}

TEST(LfmThoughtDemotion, MentionOfPlanWithoutShapeKept) {
    EXPECT_FALSE(ovms::lfm2::shouldDemoteSegment("I have a plan and an explanation ready."));
}

TEST(LfmThoughtDemotion, EmptyAndWhitespaceKept) {
    EXPECT_FALSE(ovms::lfm2::shouldDemoteSegment(""));
    EXPECT_FALSE(ovms::lfm2::shouldDemoteSegment("   \n\t"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run (Windows, from repo root, using the same standalone/focused pattern as retainer tests if full Bazel is heavy):

```powershell
# Prefer the existing focused output-parser binary if already built, e.g.:
#   ovms_test --gtest_filter=LfmThoughtDemotion.*
# Or compile via the tool-use runner style used for lfm_retainer_test.
```

Expected: compile failure (`thought_demotion.hpp` missing) or link/fail on missing `shouldDemoteSegment`.

- [ ] **Step 3: Implement minimal classifier**

Create `src/llm/io_processing/lfm2/thought_demotion.hpp` with this complete header:

```cpp
#pragma once
#include <array>
#include <cctype>
#include <string_view>

namespace ovms {
namespace lfm2 {

inline constexpr size_t kDemotionLeadingProseLimit = 200;

inline bool isAsciiSpace(unsigned char c) { return std::isspace(c) != 0; }

inline std::string_view trimView(std::string_view s) {
    while (!s.empty() && isAsciiSpace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && isAsciiSpace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

inline bool isDemotionKey(std::string_view key) {
    static constexpr std::array<std::string_view, 5> keys{
        "explanation", "plan", "commands", "command", "steps"};
    for (const auto candidate : keys)
        if (key == candidate) return true;
    return false;
}

// Quote/brace-aware scan: true if any depth-1 object key matches the demotion set.
inline bool scanDemotionKeys(std::string_view objectBody) {
    int depth = 0;
    bool inString = false;
    char quote = 0;
    bool escaped = false;
    bool expectKey = false;
    size_t keyOpen = std::string_view::npos;
    for (size_t i = 0; i < objectBody.size(); ++i) {
        const char c = objectBody[i];
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == quote) {
                inString = false;
                if (expectKey && depth == 1 && keyOpen != std::string_view::npos && quote == '"') {
                    const auto key = objectBody.substr(keyOpen + 1, i - keyOpen - 1);
                    if (isDemotionKey(key)) return true;
                    expectKey = false;
                }
                keyOpen = std::string_view::npos;
            }
            continue;
        }
        if (c == '"' || c == '\'') {
            inString = true;
            quote = c;
            if (expectKey && depth == 1) keyOpen = i;
            continue;
        }
        if (c == '{') {
            ++depth;
            expectKey = (depth == 1);
            continue;
        }
        if (c == '[') {
            ++depth;
            expectKey = false;
            continue;
        }
        if (c == '}' || c == ']') {
            if (depth > 0) --depth;
            expectKey = false;
            continue;
        }
        if (c == ',' && depth == 1) {
            expectKey = true;
            continue;
        }
        if (c == ':' && depth == 1) {
            expectKey = false;
            continue;
        }
    }
    return false;
}

inline bool looksLikePseudoToolProse(std::string_view segment) {
    segment = trimView(segment);
    if (segment.empty()) return false;
    size_t lineStart = 0;
    bool sawToolLine = false;
    for (size_t i = 0; i <= segment.size(); ++i) {
        if (i < segment.size() && segment[i] != '\n') continue;
        auto line = trimView(segment.substr(lineStart, i - lineStart));
        lineStart = i + 1;
        if (line.empty()) continue;
        if (line.size() >= 6 && line.substr(0, 6) == "[tool:") {
            if (line.find(']') != std::string_view::npos) {
                sawToolLine = true;
                continue;
            }
        }
        return false;
    }
    return sawToolLine;
}

inline bool looksLikePlanJson(std::string_view segment) {
    segment = trimView(segment);
    const auto brace = segment.find('{');
    if (brace == std::string_view::npos || brace > kDemotionLeadingProseLimit) return false;
    return scanDemotionKeys(segment.substr(brace));
}

inline bool shouldDemoteSegment(std::string_view segment) {
    segment = trimView(segment);
    if (segment.empty()) return false;
    return looksLikePseudoToolProse(segment) || looksLikePlanJson(segment);
}

}  // namespace lfm2
}  // namespace ovms
```

**Implementer requirement:** Replace the incomplete `scanDemotionKeys` sketch with a correct key-open tracking implementation before claiming green. The tests define behavior; the sketch must not ship with `return false` stubs.

Add the header to `src/llm/BUILD` on library `io_processing_lfm2_tool_parser`:

```python
hdrs = [
    "io_processing/lfm2/lfm2_tool_parser.hpp",
    "io_processing/lfm2/python_literal.hpp",
    "io_processing/lfm2/thought_demotion.hpp",
],
```

- [ ] **Step 4: Run classifier tests to verify they pass**

```powershell
# ovms_test --gtest_filter=LfmThoughtDemotion.*
```

Expected: all `LfmThoughtDemotion.*` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/io_processing/lfm2/thought_demotion.hpp src/test/llm/output_parsers/lfm_thought_demotion_test.cpp src/llm/BUILD
git commit -m "$(cat <<'EOF'
feat(lfm): add Hybrid C thought demotion classifier

EOF
)"
```

---

### Task 2: Buffer and demote in registry retainer

**Files:**
- Modify: `src/llm/io_processing/lfm2/lfm2_tool_parser.hpp`
- Modify: `src/llm/io_processing/lfm2/lfm2_tool_parser.cpp`
- Modify: `src/test/llm/output_parsers/lfm_retainer_test.cpp` (only if a content-only case asserts immediate flush; adjust expectations for demotion-off default)

**Interfaces:**
- Consumes: `ovms::lfm2::shouldDemoteSegment`
- Produces: `Lfm2ToolParser(..., bool parallelToolCalls, bool demotePseudoThought=false, optional config)`; when demote on, non-envelope text becomes ReasoningDelta or ContentDelta at segment close.

- [ ] **Step 1: Write failing retainer-level tests for demotion flag**

In `lfm_retainer_test.cpp` (or a small new section), add tests that construct the registry parser with `demotePseudoThought=true` and feed a finished plan-JSON segment with no envelope. Assert `ReasoningDelta` text equals the segment and no `ToolCallDelta`. Mirror with `demotePseudoThought=false` expecting `ContentDelta`.

Example shape (adapt to existing retainer test helpers):

```cpp
TEST(LfmRetainerDemotion, PlanJsonBecomesReasoningWhenEnabled) {
    // Construct Lfm2ToolParser(tokenizer, schemas, true, true /*demote*/);
    // parseChunk(planJson, {}, STOP); drain deltas
    // EXPECT reasoning == planJson; content empty; toolCalls empty
}

TEST(LfmRetainerDemotion, PlanJsonStaysContentWhenDisabled) {
    // demote=false (default)
    // EXPECT content == planJson; reasoning empty
}
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: compile failure on unknown ctor arg and/or content still emitted when demote=true.

- [ ] **Step 3: Extend parser state and flush helper**

In `lfm2_tool_parser.hpp`:

- Add ctor parameter `bool demotePseudoThought = false` after `parallelToolCalls` (before `configOverride`).
- Store `bool demotePseudoThought{false};`
- Add `std::string demotionBuffer;`
- In `resetState()`, clear `demotionBuffer`.
- Declare `void flushDemotionBuffer();` private.

In `lfm2_tool_parser.cpp`:

```cpp
#include "thought_demotion.hpp"

void Lfm2ToolParser::flushDemotionBuffer() {
    if (demotionBuffer.empty()) return;
    std::string segment = std::move(demotionBuffer);
    demotionBuffer.clear();
    cutEOSFromContent(segment);
    if (segment.empty()) return;
    if (demotePseudoThought && lfm2::shouldDemoteSegment(segment)) {
        pendingDeltas.emplace_back(ReasoningDelta{std::move(segment)});
    } else {
        pendingDeltas.emplace_back(ContentDelta{std::move(segment)});
    }
}
```

In `parseRetainedChunk`, replace direct `ContentDelta` emplaces for non-envelope text with:

- If `demotePseudoThought`: append to `demotionBuffer` instead of `pendingDeltas`.
- Before entering a new envelope (`retainedEnvelope = true`) and on `finished`: call `flushDemotionBuffer()`.
- Keep byte-limit checks covering `streamingContent.size() + demotionBuffer.size() + chunk.size()`.

Do **not** change `retainEnvelope` validation or ToolCallDelta publication.

- [ ] **Step 4: Run retainer demotion tests + existing retainer suite**

```powershell
# ovms_test --gtest_filter=LfmRetainer*:LfmRetainerDemotion.*
```

Expected: prior retainer tests still PASS with default demote=false; new demotion tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/io_processing/lfm2/lfm2_tool_parser.hpp src/llm/io_processing/lfm2/lfm2_tool_parser.cpp src/test/llm/output_parsers/lfm_retainer_test.cpp
git commit -m "$(cat <<'EOF'
feat(lfm): demote pseudo-thought segments in retainer content path

EOF
)"
```

---

### Task 3: Enable demotion from OutputParser (LFM2.5 only)

**Files:**
- Modify: `src/llm/io_processing/output_parser.cpp`
- Modify: `src/test/llm/output_parsers/lfm_strict_output_parser_test.cpp`

**Interfaces:**
- Consumes: Task 2 ctor `demotePseudoThought`
- Produces: production path `OutputParser(..., "lfm2", "lfm2", schemas)` enables demotion; `OutputParser(..., "lfm2", "", schemas)` does not.

- [ ] **Step 1: Write failing integrated strict tests**

Append to `lfm_strict_output_parser_test.cpp`:

```cpp
TEST_P(LfmStrictOutputParserTest, PseudoPlanJsonGoesToReasoningNotTools) {
    if (!GetParam()) GTEST_SKIP() << "LFM2.5 demotion only";
    const std::string plan =
        R"({"explanation":"simulate","plan":["cwd"],"commands":["bash /workspace"]})";
    const auto result = run({plan});
    EXPECT_EQ(result.reasoning, plan);
    EXPECT_TRUE(result.content.empty());
    EXPECT_TRUE(result.toolCalls.empty());
}

TEST_P(LfmStrictOutputParserTest, PseudoToolProseGoesToReasoningNotTools) {
    if (!GetParam()) GTEST_SKIP() << "LFM2.5 demotion only";
    const std::string prose = "[tool:cwd] /home/user\n[tool:write] /tmp/x";
    const auto result = run({prose});
    EXPECT_EQ(result.reasoning, prose);
    EXPECT_TRUE(result.content.empty());
    EXPECT_TRUE(result.toolCalls.empty());
}

TEST_P(LfmStrictOutputParserTest, DemotedProseThenNativeEnvelopePublishesToolOnly) {
    if (!GetParam()) GTEST_SKIP() << "LFM2.5 demotion only";
    const std::string text =
        std::string(R"({"explanation":"draft","plan":[]})") +
        R"(<|tool_call_start|>[f(a=1)]<|tool_call_end|>)";
    const auto result = run({text});
    EXPECT_EQ(result.reasoning, R"({"explanation":"draft","plan":[]})");
    EXPECT_TRUE(result.content.empty());
    ASSERT_EQ(result.toolCalls.size(), 1);
    EXPECT_EQ(result.toolCalls[0].name, "f");
    EXPECT_EQ(result.toolCalls[0].arguments, R"({"a":1})");
}

TEST_P(LfmStrictOutputParserTest, Lfm2ToolOnlyKeepsPlanJsonAsContent) {
    if (GetParam()) GTEST_SKIP() << "LFM2 tool-only demotion off";
    const std::string plan = R"({"explanation":"x","plan":[]})";
    const auto result = run({plan});
    EXPECT_EQ(result.content, plan);
    EXPECT_TRUE(result.reasoning.empty());
    EXPECT_TRUE(result.toolCalls.empty());
}

TEST_P(LfmStrictOutputParserTest, ThinkTagsStillPrimaryReasoningPath) {
    if (!GetParam()) GTEST_SKIP() << "LFM2.5 reasoning only";
    const auto result = run({"<think>private</think>public answer"});
    EXPECT_EQ(result.reasoning, "private");
    EXPECT_EQ(result.content, "public answer");
}
```

Note: existing tests that expect `"beforeafter"` content remain valid (ordinary prose).

- [ ] **Step 2: Run to verify fail on LFM2.5 demotion cases**

```powershell
# ovms_test --gtest_filter=LfmStrictOutputParserTest.*Plan* :*Pseudo* :*Demoted* :*Lfm2ToolOnly* :*ThinkTags*
```

Expected: LFM2.5 plan/prose cases FAIL (plan still in content) until wiring lands.

- [ ] **Step 3: Wire OutputParser**

In `output_parser.cpp`, change the `lfm2` tool branch to:

```cpp
} else if (toolParserName == "lfm2") {
    const bool demotePseudoThought = (reasoningParserName == "lfm2");
    toolParser = std::make_unique<Lfm2ToolParser>(
        tokenizer, toolNameSchemaMap, parallelToolCalls, demotePseudoThought);
}
```

Ensure the reasoning parser is still created afterward as today (`reasoningParserName == "lfm2"` → `Lfm25ReasoningParser`). Order already constructs tool parser before reasoning parser; reading `reasoningParserName` string is enough (do not require `reasoningParser` unique_ptr yet).

- [ ] **Step 4: Run strict LFM + thought demotion + retainer suites**

```powershell
# ovms_test --gtest_filter=LfmStrictOutputParserTest.*:LfmThoughtDemotion.*:LfmRetainer*
```

Expected: PASS, including both tokenizer params (LFM2 skips demotion cases; LFM2.5 exercises them). Confirm `ReasoningIsSeparatedFromContent` and envelope split tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/io_processing/output_parser.cpp src/test/llm/output_parsers/lfm_strict_output_parser_test.cpp
git commit -m "$(cat <<'EOF'
feat(lfm): enable Hybrid C demotion when reasoning_parser=lfm2

EOF
)"
```

---

### Task 4: Streaming splits + docs/state freeze

**Files:**
- Modify: `src/test/llm/output_parsers/lfm_strict_output_parser_test.cpp` (split coverage)
- Modify: `docs/lfm/tool-use-20261027/STATE.md` (brief status)
- Optional note in `docs/lfm/tool-use-20261027/FINAL-REPORT.md` outstanding list (only if editing that file anyway)

**Interfaces:**
- Consumes: Tasks 1–3 behavior
- Produces: evidence that demotion does not publish early content across splits; STATE points at design/plan.

- [ ] **Step 1: Add two-way split test for demotion buffer**

```cpp
TEST_P(LfmStrictOutputParserTest, PlanJsonTwoWaySplitsStayReasoning) {
    if (!GetParam()) GTEST_SKIP() << "LFM2.5 demotion only";
    const std::string plan =
        R"({"explanation":"need tools","plan":["cwd"],"commands":["bash /workspace"]})";
    for (size_t split = 0; split <= plan.size(); ++split) {
        SCOPED_TRACE(split);
        const auto result = run({plan.substr(0, split), plan.substr(split)});
        EXPECT_EQ(result.reasoning, plan);
        EXPECT_TRUE(result.content.empty());
        EXPECT_TRUE(result.toolCalls.empty());
    }
}
```

- [ ] **Step 2: Run split test red/green**

```powershell
# ovms_test --gtest_filter=LfmStrictOutputParserTest.PlanJsonTwoWaySplitsStayReasoning
```

Expected: PASS for LFM2.5 param.

- [ ] **Step 3: Update STATE.md**

Ensure `docs/lfm/tool-use-20261027/STATE.md` notes Design C approved and links:

- Spec: `docs/superpowers/specs/2026-10-27-lfm25-hybrid-thought-parser-design.md`
- Plan: `docs/superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md`

- [ ] **Step 4: Full focused LFM output-parser regression**

```powershell
# ovms_test --gtest_filter=Lfm*:LFM2*:LFM25*
```

Expected: no new failures vs pre-change baseline for non-demotion cases.

- [ ] **Step 5: Commit**

```bash
git add src/test/llm/output_parsers/lfm_strict_output_parser_test.cpp docs/lfm/tool-use-20261027/STATE.md
git commit -m "$(cat <<'EOF'
test(lfm): cover Hybrid C demotion splits and record design paths

EOF
)"
```

---

## Plan self-review

1. **Spec coverage:** Goals 1–4 → Tasks 1–3. Streaming/segment close → Task 2+4. LFM2 demotion off → Task 3 `Lfm2ToolOnlyKeepsPlanJsonAsContent`. Think primary path → Task 3 think test. Tool invariant → demoted-then-envelope test. Config gate → OutputParser wiring. Byte limit / resetState → Task 2.
2. **Placeholders:** Task 1 ships a complete `thought_demotion.hpp` body; no TBD/stub returns.
3. **Type consistency:** `shouldDemoteSegment(std::string_view)`, ctor `bool demotePseudoThought`, deltas `ReasoningDelta` / `ContentDelta` match across tasks.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-27-lfm25-hybrid-thought-parser.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in-session with executing-plans checkpoints  

Which approach?
