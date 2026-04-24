#!/usr/bin/env python3
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Measure translation unit sizes by extracting compile commands from
bazel aquery and re-running the preprocessor (-E -P).

This gives accurate preprocessed sizes because it uses the exact same
include paths, defines, and flags that bazel uses for the real build.

Prerequisites:
  - Must run where bazel is available (e.g. inside the -build container)
  - A successful 'bazel build //src:ovms' should have been done first
    so that generated headers (.pb.h etc.) and external repos exist.

Usage:
  python3 tools/measure_tu_sizes.py [output.csv] [bazel_target]

  output.csv    : Output CSV (default: tu_sizes.csv)
  bazel_target  : Target to analyze (default: //src/...)
"""

import json
import subprocess
import sys
import os
import re


def get_execroot():
    result = subprocess.run(
        ["bazel", "info", "execution_root"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: 'bazel info execution_root' failed: {result.stderr}",
              file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_compile_actions(target):
    """Run bazel aquery and return parsed JSON.
    We query the target directly (no deps() expansion) so that only
    compile actions for OVMS source files are returned, not for
    transitive external dependencies (gRPC, protobuf, etc.)."""
    cmd = [
        "bazel", "aquery",
        f'mnemonic("CppCompile", {target})',
        "--output=jsonproto",
        "--keep_going",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"ERROR: bazel aquery produced no output:\n{result.stderr}",
              file=sys.stderr)
        sys.exit(1)
    if result.returncode != 0:
        print(f"WARNING: bazel aquery returned non-zero but produced output, "
              f"continuing...", file=sys.stderr)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse aquery JSON: {e}", file=sys.stderr)
        sys.exit(1)


def make_preprocess_args(arguments):
    """
    Modify a CppCompile argument list to preprocess instead of compile.
    Returns (source_file, modified_args) or (None, None) on failure.
    """
    source = None
    new_args = []
    skip_next = False

    for i, arg in enumerate(arguments):
        if skip_next:
            skip_next = False
            continue

        # Replace -c with -E -P (preprocess only, no line markers)
        if arg == "-c":
            new_args.extend(["-E", "-P"])
            continue

        # Drop -o <output> since we want stdout
        if arg == "-o":
            skip_next = True
            continue

        # Drop -MD / -MF / dependency-file related flags
        if arg in ("-MD", "-MMD"):
            continue
        if arg in ("-MF", "-MQ", "-MT"):
            skip_next = True
            continue

        new_args.append(arg)

        # Identify source file
        if not arg.startswith("-") and re.search(r"\.(cpp|cc|cxx|c)$", arg):
            source = arg

    if not source:
        return None, None

    return source, new_args


def normalize_path(path):
    """Normalize to workspace-relative path like src/foo.cpp."""
    # Handle paths like: external/.../src/foo.cpp or just src/foo.cpp
    match = re.search(r"(src/.+\.(cpp|cc|cxx|c))$", path)
    if match:
        return match.group(1)
    # Fallback: strip common prefixes
    for prefix in ("./", "bazel-out/", "../"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def preprocess_and_measure(args, execroot):
    """Run the preprocessor and return (line_count, byte_count)."""
    try:
        result = subprocess.run(
            args,
            cwd=execroot,
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            return 0, 0
        pp_lines = result.stdout.count(b"\n")
        pp_bytes = len(result.stdout)
        return pp_lines, pp_bytes
    except (subprocess.TimeoutExpired, OSError) as e:
        return 0, 0


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "tu_sizes.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else "//src/..."

    print(f"=== Translation Unit Size Measurement ===")
    print(f"Target: {target}")
    print(f"Output: {output}")
    print()

    execroot = get_execroot()
    print(f"Execroot: {execroot}")

    print("Running bazel aquery (analysis only, no build)...")
    data = get_compile_actions(target)

    actions = data.get("actions", [])
    print(f"Found {len(actions)} total actions")

    # Filter to only CppCompile (aquery mnemonic filter should handle this,
    # but double-check in case of format differences)
    compile_actions = [
        a for a in actions
        if a.get("mnemonic") == "CppCompile"
    ]
    print(f"CppCompile actions: {len(compile_actions)}")
    print()

    results = []
    failures = []

    for i, action in enumerate(compile_actions):
        arguments = action.get("arguments", [])
        if not arguments:
            continue

        source, pp_args = make_preprocess_args(arguments)
        if not source:
            continue

        # Skip external dependency sources (safety net)
        if "external/" in source:
            continue

        src_path = normalize_path(source)

        # Skip test files
        if "/test/" in src_path or "test_" in os.path.basename(src_path):
            continue

        pp_lines, pp_bytes = preprocess_and_measure(pp_args, execroot)

        if pp_lines == 0:
            failures.append(src_path)
        else:
            results.append((src_path, pp_lines, pp_bytes))

        done = len(results) + len(failures)
        if done % 500 == 0:
            print(f"  Processed {done} files...", flush=True)

    # Write CSV
    with open(output, "w") as f:
        f.write("file,pp_lines,pp_bytes\n")
        for src, lines, nbytes in sorted(results):
            f.write(f"{src},{lines},{nbytes}\n")

    # Summary
    if not results:
        print("\nWARNING: No files were successfully preprocessed!")
        print("Make sure you have done a successful build first.")
        if failures:
            print(f"Failed files ({len(failures)}):")
            for fp in failures[:10]:
                print(f"  {fp}")
        return

    total_lines = sum(r[1] for r in results)
    total_bytes = sum(r[2] for r in results)
    avg_lines = total_lines // len(results)
    lines_sorted = sorted(r[1] for r in results)
    median_lines = lines_sorted[len(lines_sorted) // 2]
    max_entry = max(results, key=lambda r: r[1])

    print()
    print(f"=== Summary ===")
    print(f"Files measured:     {len(results)}")
    if failures:
        print(f"Files failed:       {len(failures)}")
    print(f"Total PP lines:     {total_lines:>12,}")
    print(f"Total PP bytes:     {total_bytes:>12,}  ({total_bytes/1024/1024:.1f} MB)")
    print(f"Average PP lines:   {avg_lines:>12,}")
    print(f"Median PP lines:    {median_lines:>12,}")
    print(f"Max PP lines:       {max_entry[1]:>12,}  ({max_entry[0]})")

    print()
    print(f"Distribution:")
    thresholds = [
        (200000, ">200K"),
        (100000, "100K-200K"),
        (50000,  "50K-100K"),
        (20000,  "20K-50K"),
        (0,      "0-20K"),
    ]
    for thresh, label in thresholds:
        if thresh == 0:
            count = sum(1 for r in results if r[1] <= 20000)
        elif thresh == 200000:
            count = sum(1 for r in results if r[1] > 200000)
        else:
            lo = thresh
            hi = thresholds[thresholds.index((thresh, label)) - 1][0]
            count = sum(1 for r in results if lo < r[1] <= hi)
        print(f"  {count:>6} files with {label:>10} lines")

    print()
    print("Top 10 largest translation units:")
    top = sorted(results, key=lambda r: r[1], reverse=True)[:10]
    for src, lines, nbytes in top:
        print(f"  {src:<55} {lines:>10,} lines  {nbytes/1024:>8,.0f} KB")

    print()
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
