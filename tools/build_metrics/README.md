# OVMS Build Metrics Tools

Tools for measuring and comparing OVMS build-graph and translation-unit
complexity. Five metrics capture different aspects of build cost and
incremental-rebuild exposure.

All scripts use **Python 3 standard library only** — no venv or pip
install needed. They do require **Bazel** to be available (i.e. they
run inside a `-build` container where the repo is mounted at `/ovms`).

### Quick Reference

| Metric | Script | Output CSVs |
|--------|--------|-------------|
| **M1** — Dependency Fan-In | `tools/build_metrics/measure_dep_fanin.py` | `deps.csv` |
| **M2** — Preprocessed TU Size | `tools/build_metrics/measure_tu_sizes.py` | `tu.csv`, `tu_src.csv` |
| **M3** — Transitive Source Files | `tools/build_metrics/measure_build_graph.py` | `trans_srcs.csv` |
| **M4** — Change Impact | `tools/build_metrics/measure_build_graph.py` | `impact.csv` |
| **M5** — Header Fan-In | `tools/build_metrics/measure_header_fanin.py` | `headers.csv`, `headers_src.csv` |
| Compare | `tools/build_metrics/compare_build_metrics.py` | *(stdout report)* |

---

## Metrics

### M1 — Dependency Fan-In (Bazel targets)

| | |
|---|---|
| **What it measures** | Number of transitive Bazel targets each source file depends on. |
| **Why it matters** | Proxy for build-graph complexity. A file with many target deps is coupled to a large part of the graph; any change in those targets triggers a recompile. |
| **Limitation** | Inflated by target splits — splitting one library into several mechanically increases dep count for all consumers even though actual build work is unchanged. Use M3 as a complementary metric. |
| **Tool** | `tools/build_metrics/measure_dep_fanin.py` |
| **Method** | Single `bazel query --output=xml` of all rules. Builds the dependency graph in memory and computes transitive closure per target in Python. |
| **Output CSV** | `deps.csv` — columns: `file, target, internal_deps, openvino, genai, tensorflow, mediapipe, grpc, protobuf, other_external, total_deps` |
| **Buckets** | 0–10, 11–20, 21–30, 31–40, 41–50, 51–60, 61–70, 71–80, 81–90, 91–100, 101–200, ≥201 deps |

### M2 — Preprocessed Translation-Unit Size

| | |
|---|---|
| **What it measures** | Number of lines and bytes after the C preprocessor expands all `#include`s and macros for each source file. |
| **Why it matters** | Direct proxy for frontend compile cost — the compiler must parse every preprocessed line. Smaller TUs compile faster, especially under Bazel parallelism. |
| **Tool** | `tools/build_metrics/measure_tu_sizes.py` |
| **Method** | Runs `bazel aquery` to extract real compile commands, then re-runs each with `-E -P` (preprocess only) and counts output. |
| **Prerequisite** | `bazel build //src:ovms` must have succeeded once so that generated headers (`.pb.h` etc.) exist. |
| **Output CSV** | `tu.csv` — columns: `file, pp_lines, pp_bytes` |
| **Buckets** | 0–10k, 10k–25k, 25k–50k, 50k–100k, 100k–200k, 200k–300k, ≥300k lines |

### M3 — Transitive Source Files

| | |
|---|---|
| **What it measures** | For each `.cpp` file, counts the total number of other `.cpp`/`.cc` files that belong to its Bazel target **and all the targets it transitively depends on**. In other words: if you do a clean build of this file's target, how many `.cpp` files need to be compiled (including itself and all its transitive library dependencies)? |
| **Why it matters** | This is a direct proxy for the size of a file's dependency subtree in terms of real compilation work. A file with `trans_src_count=222` means its target pulls in 222 source files through the chain of `deps = [...]` in BUILD files. Reducing this number (by breaking deps, splitting libraries, or removing unnecessary deps) directly shrinks what needs to be compiled. Unlike M1 which counts Bazel *targets*, this counts actual `.cpp` files, so it is immune to target-split artifacts where splitting one library into two increases the target count without changing real work. |
| **Example** | `pipelinedefinition.cpp` belonged to a target that depended (transitively) on 222 source files before optimization. After refactoring deps, it depends on only 51 — meaning 171 fewer `.cpp` files are in its compilation chain. |
| **Tool** | `tools/build_metrics/measure_build_graph.py` (first output) |
| **Method** | Single `bazel query --output=xml` of `kind("cc_library|cc_binary|cc_test", //src/...)`. Builds forward dependency graph, traverses closures, counts `.cpp`/`.cc` source files per target. |
| **Output CSV** | `trans_srcs.csv` — columns: `file, target, trans_src_count` |
| **Buckets** | 0–20, 21–50, 51–100, 101–200, 201–500, ≥501 source files |

### M4 — Change Impact (Reverse Deps)

| | |
|---|---|
| **What it measures** | How many source files would need to recompile if a given file's Bazel target is modified. |
| **Why it matters** | Measures the incremental-build "blast radius". High-impact files (e.g. `logging.cpp`, `status.cpp`) are bottleneck nodes — any change to them or their targets forces a large rebuild. Identifies where decoupling efforts have the highest ROI. |
| **Tool** | `tools/build_metrics/measure_build_graph.py` (second output) |
| **Method** | Same Bazel query as M3. Builds reverse dependency graph, traverses closures, counts impacted `.cpp`/`.cc` source files per target. |
| **Output CSV** | `impact.csv` — columns: `file, target, rdep_targets, impacted_srcs` |
| **Buckets** | 0–50, 51–100, 101–200, 201–500, ≥501 impacted source files |

### M5 — Header Fan-In (Unique Header Count)

| | |
|---|---|
| **What it measures** | Total number of unique header files resolved by the compiler for each translation unit, as well as the subset that are project headers (from `src/`). |
| **Why it matters** | Captures include-chain complexity at the compiler level (not Bazel level). A file pulling in 2000 unique headers has vastly more surface exposed to invalidation and longer preprocess/parse time than one pulling in 100. |
| **Tool** | `tools/build_metrics/measure_header_fanin.py` |
| **Method** | Runs `bazel aquery` to extract compile commands, then re-runs each with the `-M` flag (produce Makefile-format dependency list). Parses the output to count unique header paths. |
| **Prerequisite** | `bazel build //src:ovms` must have succeeded once so that generated headers exist. Without this, many files will fail. |
| **Output CSV** | `headers.csv` — columns: `file, unique_headers, unique_project_headers` |
| **Buckets** | 0–100, 101–200, 201–500, 501–1000, 1001–2000, ≥2001 headers |

---

## Comparison Tool

`tools/build_metrics/compare_build_metrics.py` — Compares before/after CSVs for any
combination of the 5 metrics and produces a report with:
- Headline stats (file count, total, average, median, P90, max)
- Bucketed distributions (count and delta per bucket)
- **Top 10 reductions** (files that improved the most)
- **Top 10 worst remaining** (files with the highest values after changes — these are the best candidates for future optimization)
- Summary verdict per metric

Supports file scopes: `all`, `non-test`, `test` (via `--scopes`).

All flags are optional — you can compare any subset of metrics:

```
--deps-before / --deps-after           M1
--tu-before / --tu-after               M2
--trans-srcs-before / --trans-srcs-after  M3
--impact-before / --impact-after       M4
--headers-before / --headers-after     M5
--scopes all,non-test,test
```

This script is **pure Python** with no Bazel dependency — it can run on
the host, in a container, or in CI.

---

## Prerequisites

- **Measurement scripts** (M1–M5) require **Bazel** and must run inside
  a `-build` container where the repo is mounted at `/ovms`.
- **M2 and M5** additionally require a prior successful
  `bazel build //src:ovms` so that generated headers (`.pb.h` etc.)
  and external repos are present. Without this, many compile actions
  will fail during measurement.
- **All scripts** use only Python 3 standard library — no virtual
  environment or third-party packages are needed.
- The comparison script (`compare_build_metrics.py`) has no Bazel
  dependency and runs anywhere with Python 3.

---

## Usage

### Option A — Makefile targets (recommended)

Run all 5 metrics inside an existing `-build` container. By default,
CSVs are written to `tools/build_metrics/current/` which is committed
to the repository — after running, `git diff` shows exactly what changed.

```bash
# Measure (writes to tools/build_metrics/current/ by default)
make build_metrics BUILD_CONTAINER=u24a_ovms1

# Compare current vs pre-optimization baseline
make build_metrics_compare

# See per-file changes in git
git diff tools/build_metrics/current/
```

You can override the output directory if needed:
```bash
make build_metrics BUILD_CONTAINER=u24a_ovms1 METRICS_DIR=tools/build_metrics/my_branch
```

The `build_metrics` target:
1. Runs all 5 measurement scripts via `docker exec`
2. Writes all output CSVs directly to `$(METRICS_DIR)/` (bind-mounted)
3. Filters TU and headers CSVs to `src/` rows only (`tu_src.csv`, `headers_src.csv`)

### Option B — Run manually inside the build container

```bash
# Inside the -build container, repo mounted at /ovms
cd /ovms

# M1: Dependency fan-in
python3 tools/build_metrics/measure_dep_fanin.py deps.csv

# M2: Preprocessed TU sizes (~5-30 min)
python3 tools/build_metrics/measure_tu_sizes.py tu.csv '//src:ovms'

# M3+M4: Transitive source files + change impact (~2 min)
python3 tools/build_metrics/measure_build_graph.py trans_srcs.csv impact.csv

# M5: Header fan-in (~10 min)
python3 tools/build_metrics/measure_header_fanin.py headers.csv '//src:ovms'

# Filter TU to src/ rows only
head -1 tu.csv > tu_src.csv
grep '^src/' tu.csv >> tu_src.csv

# Compare
python3 tools/build_metrics/compare_build_metrics.py \
  --deps-before deps_before.csv    --deps-after deps.csv \
  --tu-before tu_before_src.csv    --tu-after tu_src.csv \
  --trans-srcs-before trans_srcs_before.csv --trans-srcs-after trans_srcs.csv \
  --impact-before impact_before.csv --impact-after impact.csv \
  --headers-before headers_before.csv --headers-after headers.csv \
  --scopes all,non-test
```

### Option C — Drive from host via docker exec (bind-mounted repo)

```bash
CONTAINER=u24a_ovms1
OUT=tools/build_metrics/my_branch
mkdir -p $OUT

# M1
docker exec $CONTAINER bash -c \
  "cd /ovms && python3 tools/build_metrics/measure_dep_fanin.py /ovms/$OUT/deps.csv"

# M2
docker exec $CONTAINER bash -c \
  "cd /ovms && python3 tools/build_metrics/measure_tu_sizes.py /ovms/$OUT/tu.csv '//src:ovms'"

# M3+M4
docker exec $CONTAINER bash -c \
  "cd /ovms && python3 tools/build_metrics/measure_build_graph.py /ovms/$OUT/trans_srcs.csv /ovms/$OUT/impact.csv"

# M5
docker exec $CONTAINER bash -c \
  "cd /ovms && python3 tools/build_metrics/measure_header_fanin.py /ovms/$OUT/headers.csv '//src:ovms'"

# Filter to src/ rows only
head -1 $OUT/tu.csv > $OUT/tu_src.csv
grep '^src/' $OUT/tu.csv >> $OUT/tu_src.csv || true
head -1 $OUT/headers.csv > $OUT/headers_src.csv
grep '^src/' $OUT/headers.csv >> $OUT/headers_src.csv || true

python3 tools/build_metrics/compare_build_metrics.py \
  --deps-before baseline/deps.csv         --deps-after $OUT/deps.csv \
  --tu-before baseline/tu_src.csv         --tu-after $OUT/tu_src.csv \
  --trans-srcs-before baseline/trans_srcs.csv --trans-srcs-after $OUT/trans_srcs.csv \
  --impact-before baseline/impact.csv     --impact-after $OUT/impact.csv \
  --headers-before baseline/headers_src.csv --headers-after $OUT/headers_src.csv \
  --scopes all,non-test
```

### Option D — Compare only (no rebuild needed)

If you already have before/after CSVs, comparison runs on any machine
with Python 3:

```bash
# Point to two directories — auto-resolves all CSV filenames
python3 tools/build_metrics/compare_build_metrics.py \
  --before-dir tools/build_metrics/prebuild_baseline \
  --after-dir tools/build_metrics/current \
  --scopes all,non-test,test

# Or specify individual files (any subset works)
python3 tools/build_metrics/compare_build_metrics.py \
  --trans-srcs-before before/trans_srcs.csv \
  --trans-srcs-after after/trans_srcs.csv \
  --scopes non-test
```

---

## Committed CSVs and git-diff workflow

Two sets of CSVs are committed in the repository:

| Directory | Contents |
|-----------|----------|
| `tools/build_metrics/current/` | Latest metrics for the current branch. Overwritten by `make build_metrics`. |
| `tools/build_metrics/prebuild_baseline/` | Pre-optimization snapshot (commit `d1ff1c104`, main before build optimizations). |

After running `make build_metrics`, use `git diff tools/build_metrics/current/`
to see per-file metric changes. Commit the updated CSVs with your PR so
reviewers can see the impact.

To create a new baseline on a different branch:

```bash
make build_metrics BUILD_CONTAINER=u24a_ovms_ref METRICS_DIR=tools/build_metrics/new_baseline
```

---

## Interpreting the outputs

**Key insight:** Clean build time is often unchanged by refactoring, but
**incremental builds are where developers feel the impact** — in local
edit-test cycles, precommit checks, and CI pipelines that reuse caches.

| Metric | What improving it means |
|--------|------------------------|
| **M1 (dep fan-in)** | Fewer Bazel targets in the transitive closure → more granular graph, better incremental invalidation. |
| **M2 (PP lines)** | Smaller translation units → less parsing work per compile, faster frontend. |
| **M3 (trans source files)** | Fewer real `.cpp` files in the transitive closure → same benefit as M1 but immune to target-split artifacts. |
| **M4 (change impact)** | Smaller blast radius → editing a file triggers fewer recompiles across the project. |
| **M5 (header fan-in)** | Fewer unique headers per TU → less include-chain complexity, faster preprocessing, less invalidation surface. |

For incremental build verification, measure recompilation time after a
single-file change. For absolute clean-build impact, verify with Bazel
timing or profile output.

---

## Sample Output

### `make build_metrics` — per-metric console output

Each measurement script prints a summary to stdout. Example excerpts:

**M1 (Dependency Fan-In):**
```
=== Summary ===
Files:                318
Unique targets:       236
Avg total deps/file:  60.2
Avg internal deps:    42.9
Median total deps:    22
Max total deps:       327  (src/test/openvino_remote_tensors_tests.cpp)

Distribution:
      37 files with     >200 deps
       5 files with  101-200 deps
      41 files with   51-100 deps
      80 files with    21-50 deps
     155 files with     0-20 deps
```

**M2 (Preprocessed TU Size):**
```
=== Summary ===
Files measured:     4037
Total PP lines:      312,083,910
Average PP lines:         77,305
Max PP lines:            282,053  (src/server.cpp)

Top 10 largest translation units:
  src/server.cpp                          282,053 lines    11,263 KB
  src/capi_frontend/capi.cpp              281,468 lines    11,412 KB
  src/prediction_service.cpp              275,939 lines    11,124 KB
  ...
```

**M3+M4 (Transitive Source Files + Change Impact):**
```
=== Summary: Transitive Source Files ===
  Files:    409
  Average:  116.6
  Max:      379

=== Summary: Change Impact ===
  Files:    409
  Average:  116.6
  Max:      349

  Top 10 by change impact:
    src/logging.cpp           349 impacted  (199 rdep targets)
    src/stringutils.cpp       309 impacted  (162 rdep targets)
    src/status.cpp            306 impacted  (160 rdep targets)
    ...
```

**M5 (Header Fan-In):**
```
=== Summary ===
Files measured:          4061
Avg unique headers:      396
Max unique headers:      2183

Top 10 files by unique header count:
  src/filesystem/filesystemfactory.cpp    2183 headers  (11 project)
  src/filesystem/azurefilesystem.cpp      1568 headers  (8 project)
  src/server.cpp                          1350 headers  (54 project)
  ...
```

### `make build_metrics_compare` — comparison report

The compare tool prints a structured report per metric and scope. Example excerpt (non-test scope):

```
================================================================================
  METRIC 1: Dependency Fan-in per Source File [Non-Test Files]
================================================================================

  Headline Metrics:
                                               Before      After     Change
  ------------------------------------------------------------------------
  Files measured                                  249        261
  Average total deps/file                        57.5       42.8     -25.5%
  Median total deps                                18         18
  Max total deps                                  290        327

  Top 10 Dependency Reductions:
  File                                                 Before    After   Change
  ----------------------------------------------------------------------------
  src/customloaders.cpp                                   282       14     -268
  src/global_sequences_viewer.cpp                         282       23     -259
  src/cleaner_utils.cpp                                   282       25     -257
  ...

================================================================================
  VERDICT [Non-Test Files]
================================================================================

  [IMPROVED] Dependency fan-in reduced by 25.5% on average
  [IMPROVED] High-dep files (>100): 34 -> 19 (-15)
  [IMPROVED] Total preprocessed lines reduced by 18.9%
  [IMPROVED] Trans source files reduced by 32.7% on average
  [IMPROVED] Change impact reduced by 11.1% on average
  [REGRESSED] Header fan-in increased by 2.9% on average
```

The full report includes all 5 metrics × all requested scopes (all, non-test, test), with distribution buckets, top 10 reductions, and top 10 worst remaining files per metric.

---

## Notes

- **M5 failures:** Header fan-in measurement requires all generated
  headers to exist. If `bazel build //src:ovms` has not been run, many
  compile actions will fail (the script reports the failure count).
  Always build the target first.

- **Approximate timings:**
  - M1: ~1–2 min
  - M2: ~5–30 min (depends on file count)
  - M3+M4: ~2 min

---

## Recommended KPIs for Comparing Iterations

For each optimization batch, keep the same baseline and compare.

**Primary KPIs:**
- M1: Average total deps/file, files with >100 deps
- M2: Total PP lines, average PP lines/file
- M3: Average transitive source files/TU
- M4: Average impacted sources/file

**Secondary KPIs:**
- M5: Average unique headers/file, average project headers/file
- P90/P95 deps and PP lines
- Top 10 reductions and top 10 worst remaining per metric
  - M5: ~10 min