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
Measure transitive source file count and change impact per source file.

Computes two metrics from a single Bazel XML query:

  Metric 3: Transitive Source Files (trans_src_count)
    For each source file, counts total .cpp/.cc files in the transitive
    dependency closure of its Bazel target. Unlike the target-count metric,
    this is immune to the artifact where splitting one target into several
    inflates the count without changing actual build complexity.

  Metric 4: Change Impact / Reverse Deps (impacted_srcs)
    For each source file, counts how many source files would need to
    recompile if its Bazel target is modified. Computed via reverse
    dependency graph traversal. Measures incremental build blast radius.

Output CSV 1 (trans_srcs): file, target, trans_src_count
Output CSV 2 (change_impact): file, target, rdep_targets, impacted_srcs

Usage:
  python3 tools/measure_build_graph.py [trans_srcs.csv] [change_impact.csv]

Must be run where bazel is available (e.g. inside the -build container).
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def run_bazel_query(query_expr, output_format="label"):
    """Run a bazel query and return stdout."""
    cmd = ["bazel", "query", query_expr, f"--output={output_format}",
           "--noimplicit_deps", "--keep_going"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def parse_xml_rules(xml_text):
    """
    Parse bazel query XML output.
    Returns dict: {target_label: {"srcs": [...], "deps": [...]}}
    """
    rules = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"WARNING: XML parse error: {e}", file=sys.stderr)
        return rules

    for rule in root.iter("rule"):
        label = rule.get("name", "")
        if not label:
            continue

        srcs = []
        deps = []

        for attr_list in rule.iter("list"):
            attr_name = attr_list.get("name", "")
            if attr_name == "srcs":
                for item in attr_list.iter("label"):
                    val = item.get("value", "")
                    if val:
                        srcs.append(val)
            elif attr_name in ("deps", "implementation_deps"):
                for item in attr_list.iter("label"):
                    val = item.get("value", "")
                    if val:
                        deps.append(val)

        for attr_label in rule.iter("label"):
            attr_name = attr_label.get("name", "")
            if attr_name == "deps" and attr_label.get("value"):
                deps.append(attr_label.get("value"))

        rules[label] = {"srcs": srcs, "deps": deps}

    return rules


def transitive_closure(graph, start, cache):
    """Compute set of all nodes reachable from start (including start)."""
    if start in cache:
        return cache[start]
    cache[start] = set()  # cycle guard
    result = {start}
    for neighbor in graph.get(start, []):
        result |= transitive_closure(graph, neighbor, cache)
    cache[start] = result
    return result


def count_src_files(rules, targets):
    """Count .cpp/.cc/.cxx source files across a set of targets."""
    count = 0
    for t in targets:
        if t in rules:
            count += sum(1 for s in rules[t]["srcs"]
                         if s.endswith((".cpp", ".cc", ".cxx")))
    return count


def normalize_src_path(src):
    """Convert Bazel label to workspace-relative path."""
    path = src.replace("//", "").replace(":", "/")
    if path.startswith("/"):
        path = path[1:]
    return path


def main():
    trans_output = sys.argv[1] if len(sys.argv) > 1 else "trans_srcs.csv"
    impact_output = sys.argv[2] if len(sys.argv) > 2 else "change_impact.csv"

    print("=== Build Graph Metrics (Trans Source Files + Change Impact) ===")
    print(f"Output 1 (trans source files): {trans_output}")
    print(f"Output 2 (change impact):      {impact_output}")
    print()

    sys.setrecursionlimit(10000)

    # Step 1: Get all targets (including tests — needed for rdep analysis)
    print("Step 1: Querying all targets in //src/... ...")
    target_query = 'kind("cc_library|cc_binary|cc_test", //src/...)'
    targets_text = run_bazel_query(target_query, "label")
    all_targets = [t.strip() for t in targets_text.strip().split("\n") if t.strip()]
    print(f"  Found {len(all_targets)} targets")

    # Step 2: Get rules with their deps/srcs via XML.
    # We query ONLY //src/... rules (no deps() expansion) to avoid
    # pulling in massive external dep trees. The resulting graph is
    # internal-only, which is exactly what we want for counting
    # internal source files.
    print("Step 2: Querying rule definitions (single XML query)...")
    xml_text = run_bazel_query(target_query, "xml")
    print(f"  XML response: {len(xml_text)} bytes")

    rules = parse_xml_rules(xml_text)
    print(f"  Parsed {len(rules)} rules")

    # Step 3: Build forward and reverse graphs (internal edges only)
    print("Step 3: Building dependency graphs...")
    internal_targets = set(rules.keys())
    forward_graph = {}
    reverse_graph = defaultdict(list)
    for target, info in rules.items():
        # Only follow edges to other internal (//src) targets
        internal_deps = [d for d in info["deps"] if d in internal_targets]
        forward_graph[target] = internal_deps
        for dep in internal_deps:
            reverse_graph[dep].append(target)

    # Step 4: Compute metrics per target, then expand to per-file
    print("Step 4: Computing metrics per source file...")
    forward_cache = {}
    reverse_cache = {}

    trans_results = []
    impact_results = []

    for target in sorted(all_targets):
        if target not in rules:
            continue

        cpp_srcs = [s for s in rules[target]["srcs"]
                    if s.endswith((".cpp", ".cc", ".cxx"))]
        if not cpp_srcs:
            continue

        # Forward: transitive source file count
        fwd_targets = transitive_closure(forward_graph, target, forward_cache)
        trans_src_count = count_src_files(rules, fwd_targets)

        # Reverse: change impact (blast radius)
        rev_targets = transitive_closure(reverse_graph, target, reverse_cache)
        impacted_src_count = count_src_files(rules, rev_targets)
        rdep_target_count = sum(1 for t in rev_targets
                                if t != target and t in rules)

        for src in cpp_srcs:
            filepath = normalize_src_path(src)
            trans_results.append({
                "file": filepath,
                "target": target,
                "trans_src_count": trans_src_count,
            })
            impact_results.append({
                "file": filepath,
                "target": target,
                "rdep_targets": rdep_target_count,
                "impacted_srcs": impacted_src_count,
            })

    # Step 5: Write CSVs
    print(f"\nStep 5: Writing {len(trans_results)} entries...")

    with open(trans_output, "w") as f:
        f.write("file,target,trans_src_count\n")
        for r in sorted(trans_results, key=lambda x: x["file"]):
            f.write(f"{r['file']},{r['target']},{r['trans_src_count']}\n")
    print(f"  Wrote {trans_output}")

    with open(impact_output, "w") as f:
        f.write("file,target,rdep_targets,impacted_srcs\n")
        for r in sorted(impact_results, key=lambda x: x["file"]):
            f.write(f"{r['file']},{r['target']},{r['rdep_targets']},"
                    f"{r['impacted_srcs']}\n")
    print(f"  Wrote {impact_output}")

    # Summary
    if not trans_results:
        print("\nWARNING: No results generated!")
        return

    trans_vals = sorted(r["trans_src_count"] for r in trans_results)
    impact_vals = sorted(r["impacted_srcs"] for r in impact_results)

    print()
    print("=== Summary: Transitive Source Files ===")
    print(f"  Files:    {len(trans_vals)}")
    print(f"  Average:  {sum(trans_vals) / len(trans_vals):.1f}")
    print(f"  Median:   {trans_vals[len(trans_vals) // 2]}")
    print(f"  Max:      {max(trans_vals)}")

    print()
    print("  Distribution:")
    for lo, hi, label in [
        (201, None, ">200"),
        (101, 200, "101-200"),
        (51, 100, "51-100"),
        (21, 50, "21-50"),
        (0, 20, "0-20"),
    ]:
        if hi is None:
            count = sum(1 for v in trans_vals if v > lo - 1)
        else:
            count = sum(1 for v in trans_vals if lo <= v <= hi)
        print(f"    {count:>6} files with {label:>8} trans srcs")

    print()
    non_test_trans = [r for r in trans_results if "/test/" not in r["file"]]
    print("  Top 10 by transitive source files (non-test):")
    top_trans = sorted(non_test_trans, key=lambda r: -r["trans_src_count"])[:10]
    for r in top_trans:
        f = r["file"]
        if len(f) > 55:
            f = "..." + f[-52:]
        print(f"    {f:<55s} {r['trans_src_count']:>5} srcs  ({r['target']})")

    print()
    print("=== Summary: Change Impact ===")
    print(f"  Files:    {len(impact_vals)}")
    print(f"  Average:  {sum(impact_vals) / len(impact_vals):.1f}")
    print(f"  Median:   {impact_vals[len(impact_vals) // 2]}")
    print(f"  Max:      {max(impact_vals)}")

    print()
    print("  Distribution:")
    for lo, hi, label in [
        (501, None, ">500"),
        (201, 500, "201-500"),
        (101, 200, "101-200"),
        (51, 100, "51-100"),
        (0, 50, "0-50"),
    ]:
        if hi is None:
            count = sum(1 for v in impact_vals if v > lo - 1)
        else:
            count = sum(1 for v in impact_vals if lo <= v <= hi)
        print(f"    {count:>6} files with {label:>8} impacted srcs")

    print()
    print("  Top 10 by change impact:")
    top_impact = sorted(impact_results, key=lambda r: -r["impacted_srcs"])[:10]
    for r in top_impact:
        f = r["file"]
        if len(f) > 55:
            f = "..." + f[-52:]
        print(f"    {f:<55s} {r['impacted_srcs']:>5} impacted  "
              f"({r['rdep_targets']} rdep targets)")

    print()
    print(f"Saved to: {trans_output}, {impact_output}")


if __name__ == "__main__":
    main()
