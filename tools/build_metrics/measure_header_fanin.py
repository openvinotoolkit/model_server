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
Measure header fan-in: count unique headers included per translation unit.

Uses bazel aquery to extract compile commands, then runs the compiler
with -M flag to get the exact dependency list (all headers resolved via
real include paths, defines, and conditional compilation).

Output CSV columns:
  file, unique_headers, unique_project_headers

  unique_headers:         total unique header files (project + external + system)
  unique_project_headers: headers from src/ and generated from src/ protos

Prerequisites:
  - Must run where bazel is available (e.g. inside the -build container)
  - A successful 'bazel build //src:ovms' should have been done first
    so that generated headers (.pb.h etc.) and external repos exist.

Usage:
  python3 tools/measure_header_fanin.py [output.csv] [bazel_target]
"""

import json
import os
import re
import subprocess
import sys


def get_execroot():
    """Get the bazel execution root directory."""
    result = subprocess.run(
        ["bazel", "info", "execution_root"],
        capture_output=True, text=True,
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
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse aquery JSON: {e}", file=sys.stderr)
        sys.exit(1)


def make_depfile_args(arguments):
    """
    Modify a CppCompile argument list to generate dependency info (-M).
    Returns (source_file, modified_args) or (None, None) on failure.
    """
    source = None
    new_args = []
    skip_next = False

    for i, arg in enumerate(arguments):
        if skip_next:
            skip_next = False
            continue

        # Replace -c with -M (dependency generation, no compilation)
        if arg == "-c":
            new_args.append("-M")
            continue

        # Drop -o <output>
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
    match = re.search(r"(src/.+\.(cpp|cc|cxx|c))$", path)
    if match:
        return match.group(1)
    for prefix in ("./", "bazel-out/", "../"):
        if path.startswith(prefix):
            path = path[len(prefix):]
    return path


def parse_dep_output(output_text):
    """
    Parse Makefile-format dependency output from -M flag.
    Returns set of unique dependency file paths (excluding the target .o).
    """
    # Join continuation lines (backslash-newline)
    text = output_text.replace("\\\n", " ").replace("\\\r\n", " ")

    # Split on first colon to separate target from prerequisites
    parts = text.split(":", 1)
    if len(parts) < 2:
        return set()

    prereqs_text = parts[1].strip()
    prereqs = prereqs_text.split()

    return set(prereqs)


def is_project_header(path):
    """Check if a header path is from the project (src/ or generated from src/)."""
    return (path.startswith("src/") or
            (path.startswith("bazel-out/") and "/bin/src/" in path))


def run_depfile(args, execroot):
    """Run compiler with -M and return set of dependency paths."""
    try:
        result = subprocess.run(
            args,
            cwd=execroot,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return set()
        return parse_dep_output(result.stdout)
    except (subprocess.TimeoutExpired, OSError):
        return set()


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "header_fanin.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else "//src/..."

    print("=== Header Fan-in Measurement ===")
    print(f"Target: {target}")
    print(f"Output: {output}")
    print()

    execroot = get_execroot()
    print(f"Execroot: {execroot}")

    print("Running bazel aquery...")
    data = get_compile_actions(target)

    actions = data.get("actions", [])
    compile_actions = [a for a in actions if a.get("mnemonic") == "CppCompile"]
    print(f"CppCompile actions: {len(compile_actions)}")
    print()

    results = []
    failures = []

    for action in compile_actions:
        arguments = action.get("arguments", [])
        if not arguments:
            continue

        source, dep_args = make_depfile_args(arguments)
        if not source:
            continue

        # Skip external dependency sources (safety net)
        if "external/" in source:
            continue

        src_path = normalize_path(source)

        deps = run_depfile(dep_args, execroot)

        if not deps:
            failures.append(src_path)
        else:
            # Remove the source file itself — deps includes source + all headers
            headers = {p for p in deps if p != source}
            # Also filter by basename match in case of path variations
            source_base = os.path.basename(source)
            headers = {h for h in headers
                       if os.path.basename(h) != source_base
                       or not h.endswith((".cpp", ".cc", ".cxx", ".c"))}

            unique_total = len(headers)
            unique_project = sum(1 for h in headers if is_project_header(h))

            results.append((src_path, unique_total, unique_project))

        done = len(results) + len(failures)
        if done % 500 == 0:
            print(f"  Processed {done} files...", flush=True)

    # Write CSV
    with open(output, "w") as f:
        f.write("file,unique_headers,unique_project_headers\n")
        for src, total, project in sorted(results):
            f.write(f"{src},{total},{project}\n")

    # Summary
    if not results:
        print("\nWARNING: No files were successfully measured!")
        print("Make sure you have done a successful build first.")
        if failures:
            print(f"Failed files ({len(failures)}):")
            for fp in failures[:10]:
                print(f"  {fp}")
        return

    totals = sorted(r[1] for r in results)
    projects = sorted(r[2] for r in results)

    print()
    print("=== Summary ===")
    print(f"Files measured:          {len(results)}")
    if failures:
        print(f"Files failed:            {len(failures)}")
    print(f"Avg unique headers:      {sum(totals) / len(totals):.0f}")
    print(f"Median unique headers:   {totals[len(totals) // 2]}")
    print(f"Max unique headers:      {max(totals)}")
    print()
    print(f"Avg project headers:     {sum(projects) / len(projects):.0f}")
    print(f"Median project headers:  {projects[len(projects) // 2]}")
    print(f"Max project headers:     {max(projects)}")

    print()
    print("Distribution (unique headers):")
    for lo, hi, label in [
        (2001, None, ">2000"),
        (1001, 2000, "1001-2000"),
        (501, 1000, "501-1000"),
        (201, 500, "201-500"),
        (101, 200, "101-200"),
        (0, 100, "0-100"),
    ]:
        if hi is None:
            count = sum(1 for v in totals if v > lo - 1)
        else:
            count = sum(1 for v in totals if lo <= v <= hi)
        print(f"  {count:>6} files with {label:>10} headers")

    print()
    print("Top 10 files by unique header count:")
    top = sorted(results, key=lambda r: -r[1])[:10]
    for src, total, project in top:
        if len(src) > 55:
            src = "..." + src[-52:]
        print(f"  {src:<55s} {total:>6} headers  ({project} project)")

    print()
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
