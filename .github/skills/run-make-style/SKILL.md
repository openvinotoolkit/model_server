---
name: run-make-style
description: "Use when running OVMS code-style and lint checks via the repository Makefile. The umbrella target `make style` chains spell, clang-format-check, cpplint, and cppclean. Also covers the individual sub-targets and how to fix the most common failures (uncommitted clang-format changes, cpplint warnings, cppclean unused includes, spelling/whitelist). Trigger phrases: 'run make style', 'check code style', 'cpplint', 'clang-format-check', 'cppclean', 'spell check', 'fix style', 'pre-commit style'."
---

# Run `make style` and Verify the Output

Use this skill any time the user asks to run code-style / lint checks on OVMS using the repository `Makefile`. These checks are typically run on the developer host or CI runner rather than inside the `-build` container, because the build Dockerfiles may not include `clang-format` and other style tooling. The `venv-style` target creates `.venv-style/` in whatever environment `make` is executed from using `ci/style_requirements.txt`.

## When to use

- User asks to run `make style`, `make cpplint`, `make clang-format`, `make clang-format-check`, `make cppclean`, `make spell`, or `make sdl-check`.
- User asks to "check code style", "fix style", "verify before commit / push", "run lint", "format C++ files".
- A previous style run failed and the user wants to re-run it after fixes.

## Do NOT use

- For Bazel build/test invocations — use `build-bazel-target` / `run-single-gtest`.
- For SDL / security-only checks — call `make sdl-check` directly (covered briefly below).
- For modifying source files to *suppress* legitimate cpplint/cppclean findings without addressing them.

## What `make style` actually runs

`make style` (Makefile line 283) is an umbrella that depends on:

1. **`venv-style`** — provisions `.venv-style/` and installs `ci/style_requirements.txt` (one-off per environment refresh).
2. **`spell`** — runs `codespell` over tracked + staged files, filtered through `spelling-whitelist.txt`.
3. **`clang-format-check`** — runs `clang-format -i` (target `clang-format`) on every `*.cpp/*.hpp/*.cc/*.cxx` under `src/`, then verifies the working tree and the index are clean. **It rewrites files in place** before checking — be aware.
4. **`cpplint`** — runs `cpplint` with the project options (`STYLE_CHECK_OPTS`, line length 120, the OVMS filter list) over `STYLE_CHECK_DIRS = src`.
5. **`cppclean`** — invokes `ci/cppclean.sh` to flag unused / redundant includes.

`make style` exits non-zero on the first failing step.

## Pre-flight

1. Confirm the workspace is the OVMS repo root (contains `Makefile`, `ci/style_requirements.txt`, `spelling-whitelist.txt`).
2. Confirm `python3`, `git`, and `clang-format` are available on the host.
3. **Commit or stash any in-progress edits first.** `clang-format-check` will fail the run if `clang-format` rewrites tracked files, even if the rewrites were the right thing to do — so if you intend to lint your own changes, commit them first, then run `make style`, then commit any formatting changes on top.

## Default invocation

```bash
make style
```

The first run on a clean checkout also creates `.venv-style/` and installs the requirements; subsequent runs reuse it.

## Targeted sub-targets

Run these when iterating on a specific class of failure — they're much faster than the full chain.

| Target | Purpose |
|--------|---------|
| `make spell` | Spell-check tracked + staged files via `codespell`, honoring `spelling-whitelist.txt`. |
| `make clang-format` | **Rewrite** all `src/**/*.{cpp,hpp,cc,cxx}` with the project clang-format style. |
| `make clang-format-check` | Same as above, then `git diff --exit-code` (working tree) and `git diff --exit-code --staged` (index) — **fails if it changed anything**. |
| `make cpplint` | Run cpplint only. |
| `make cppclean` | Run `ci/cppclean.sh` only (unused includes, etc.). |
| `make sdl-check` | hadolint (Dockerfiles) + bandit (Python) + license-headers (Apache 2.0). Slower; use only when the user asks for SDL. |

## Recipes

### 1. Full style pass before pushing

```bash
make style
```

### 2. Auto-format everything, then re-verify

```bash
make clang-format
git status                    # review the rewrites
git add -p                    # stage what you want
make clang-format-check       # should now exit clean
```

### 3. Iterate on cpplint / cppclean only

```bash
make cpplint
make cppclean
```

### 4. Spell-check only

```bash
make spell
```

If a legitimate term is flagged, append it to `spelling-whitelist.txt` (lowercase) **as a separate commit** rather than trying to silence the failure inline.

## Verifying the output

For each step, look for these in the make output:

| Step | Success signal | Failure signal & fix |
|------|----------------|----------------------|
| `spell` | `Spelling check completed.` | A list of `file:line: word ==> suggestion` lines. Either fix the typo or add the word to `spelling-whitelist.txt`. |
| `clang-format-check` | No diff (`make style` proceeds to `cpplint`). | `clang-format changes not committed. Commit those changes first` — run `git diff` to see what was rewritten, commit the formatting, then re-run. |
| `cpplint` | `make` proceeds to `cppclean` with no `Total errors found:` line. | Lines like `src/foo.cpp:123: <message> [<category>] [confidence]`. Address the warnings; do **not** add `// NOLINT` unless the project already does so for the same case. |
| `cppclean` | `make` finishes the umbrella step. | List of unused / redundant includes. Remove them per the OVMS include-what-you-use rule (forward-declare in headers, include in `.cpp`). |
| `make style` overall | Process exits with code `0` and no error lines preceding it. | Non-zero exit; the failing step is the last one whose output appears before the make `[Error N]` line. |

When summarizing for the user, capture:

- Which step failed (if any).
- The first 5–10 offending lines (the rest is usually repetitive).
- The concrete fix from the table above.

## Tips & pitfalls

- **`clang-format` rewrites files in place.** If you already had uncommitted edits, your edits will be intermixed with formatting changes. Commit (or stash) first, then run `make clang-format`, then commit the formatting separately.
- **`clang-format-check` depends on `clang-format`.** It is *not* a dry-run; it actually rewrites files and then asserts the working tree / index are clean. The failure message "Commit those changes first" really means "the formatter changed something — review and commit the formatting".
- **`STYLE_CHECK_DIRS = src`.** Only `src/` is checked. Code in `client/`, `demos/`, etc. is excluded from `make style`.
- **`spell` operates on tracked + staged files only.** A typo in an unstaged new file is invisible until it is at least `git add`-ed.
- **Filter list in `STYLE_CHECK_OPTS`.** Several cpplint categories are intentionally suppressed (`-build/c++11`, `-runtime/references`, `-whitespace/braces`, `-whitespace/indent`, `-build/include_order`, `-runtime/indentation_namespace`, `-build/namespaces`, `-whitespace/line_length`, `-runtime/string`, `-readability/casting`, `-runtime/explicit`, `-readability/todo`). Do not "fix" warnings that are explicitly disabled; if cpplint flags one of them, the project filter is the source of truth.
- **Line length is 120**, not 80 (set via `--linelength=120`).
- **`make style` does not run on the `-build` container.** It is host-side and relies on the host's `python3` + venv. The `-build` image is for Bazel work only.
- **License headers** are checked by `make sdl-check`, not by `make style`. New files still need the Apache 2.0 header — add it before pushing.
