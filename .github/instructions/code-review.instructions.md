---
description: "C++ review standards for OVMS pull requests and code changes"
applyTo: "src/**/*.{cc,cpp,h,hpp}"
---
# C++ Review Standards (OVMS)

Apply these when reviewing a PR or writing C++ to review quality. For include/interface
mechanics see [cpp-headers.instructions.md](cpp-headers.instructions.md) and
[cpp-sources.instructions.md](cpp-sources.instructions.md).

## Guidelines & performance

- Follow the **C++ Core Guidelines**; cite the specific guideline in review comments.
- **No hidden performance costs**: avoid `dynamic_cast` on the hot path — use `static_cast` or redesign when the type is known.
- **Avoid copies**: pass large structures (tensors, buffers) by `const&` or move them; never copy.
- Pass non-fundamental values by `const` reference wherever possible.
- Prefer member initializer lists over assignment in constructor bodies.
- When initial container values are known upfront, prefer initializer-list / brace-initialization over building an empty container and inserting.

## Correctness & safety

- **No dangling references / temporaries bound to `const T&`**:
  - Never give a `const T&` parameter a default that constructs a temporary (e.g. `const std::string& p = ""`). Use an overload or pass by value.
  - When a `const T&` is stored, verify the referenced object outlives its use.
  - Prefer overloads over default arguments for non-trivial types passed by reference.
- Verify every newly introduced non-`void` function has its result used at least once.
- No unused functions or includes (build times are long — do not add unneeded `#include`s).

## Style & readability

- No `using namespace std;` / `using namespace ov;`. Use specific `using` declarations if needed.
- Avoid `auto` for primitive types where it obscures readability.
- Use `const` and `constexpr` wherever possible.
- Descriptive names; extract duplicated logic into reusable utilities.

## Comments

Comments bloat the code and are a frequent review rejection — keep them minimal.

- Add a comment only to state what the code cannot show on its own, and keep it to one short line.
- Do not add comments where the code, function name, or test name already conveys the intent.
- No "what happens next" narration before test cases or code blocks — the name should convey intent.
- Only comment genuinely non-obvious things: workarounds, subtle ordering constraints, non-obvious rationale.
- Do not restate what the next line does or explain a change to the reviewer in the code.

## Documentation & tests

- New public APIs get docstrings in C++ headers and Python bindings; update `docs/` as needed.
- New features or behavior changes must have corresponding tests in `src/test/`.
