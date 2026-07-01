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
Fast dependency fan-in measurement using a SINGLE bazel query call.

Instead of running O(n) individual `bazel query deps()` calls,
this script:
  1. Runs ONE `bazel query --output=xml` to get all rules with their
     direct deps and srcs attributes.
  2. Builds the dependency graph in memory.
  3. Computes transitive dep counts per target in Python.

This is orders of magnitude faster than the shell-based approach.

Output CSV columns:
  file, target, internal_deps, openvino, genai, tensorflow, mediapipe,
  grpc, protobuf, other_external, total_deps

Usage:
  python3 tools/measure_dep_fanin.py [output.csv]

Must be run where bazel is available (e.g. inside the -build container).
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


# External dep groups (binary: 0 or 1 per target)
EXTERNAL_GROUPS = {
    "openvino":   ["@linux_openvino", "@windows_openvino", "//third_party:openvino"],
    "genai":      ["@linux_genai", "@windows_genai", "//third_party:genai"],
    "tensorflow": ["@org_tensorflow", "@tensorflow_serving"],
    "mediapipe":  ["@mediapipe"],
    "grpc":       ["@com_github_grpc_grpc"],
    "protobuf":   ["@com_google_protobuf"],
}


def run_bazel_query(query_expr, output_format="label"):
    """Run a bazel query and return stdout."""
    cmd = ["bazel", "query", query_expr, f"--output={output_format}",
           "--noimplicit_deps", "--keep_going"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # --keep_going may return non-zero but still produce useful output
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

        # Also check singular attrs (non-list)
        for attr_label in rule.iter("label"):
            parent = attr_label
            # label elements directly inside rule (not in list)
            attr_name = attr_label.get("name", "")
            if attr_name == "deps" and attr_label.get("value"):
                deps.append(attr_label.get("value"))

        rules[label] = {"srcs": srcs, "deps": deps}

    return rules


def compute_transitive_deps(rules, target, cache=None):
    """
    Compute the set of all transitive dependencies for a target.
    Returns a set of labels (including the target itself).
    """
    if cache is None:
        cache = {}
    if target in cache:
        return cache[target]

    # Mark as in-progress to avoid cycles
    cache[target] = set()

    result = {target}
    if target in rules:
        for dep in rules[target]["deps"]:
            child_deps = compute_transitive_deps(rules, dep, cache)
            result |= child_deps

    cache[target] = result
    return result


def classify_dep(label):
    """Classify a dependency label. Returns (group_name, is_internal)."""
    for group, prefixes in EXTERNAL_GROUPS.items():
        for prefix in prefixes:
            if label.startswith(prefix):
                return group, False

    if label.startswith("//src"):
        return "internal", True

    # Other external
    return "other_external", False


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "deps_metrics.csv"
    print(f"=== Dependency Fan-in Measurement (M1) ===")
    print(f"Output: {output}")
    print()

    # Step 1: Get all non-test targets in //src/...
    print("Step 1: Querying all targets in //src/... ...")
    targets_text = run_bazel_query(
        'kind("cc_library|cc_binary", //src/...) except kind("cc_test", //src/...)',
        "label"
    )
    all_targets = [t.strip() for t in targets_text.strip().split("\n") if t.strip()]
    print(f"  Found {len(all_targets)} targets")

    # Step 2: Get full rule definitions with deps and srcs via XML
    # Query a broad set: all rules reachable from our targets
    print("Step 2: Querying dependency graph (single XML query)...")
    xml_text = run_bazel_query(
        'deps(kind("cc_library|cc_binary", //src/...) except kind("cc_test", //src/...))',
        "xml"
    )
    print(f"  XML response: {len(xml_text)} bytes")

    rules = parse_xml_rules(xml_text)
    print(f"  Parsed {len(rules)} rules from XML")

    # Step 3: compute transitive deps per target
    print("Step 3: Computing transitive dependencies...")
    cache = {}
    results = []

    for target in sorted(all_targets):
        if target not in rules:
            continue

        srcs = rules[target]["srcs"]
        cpp_srcs = [s for s in srcs if s.endswith((".cpp", ".cc", ".cxx"))]
        if not cpp_srcs:
            continue

        trans_deps = compute_transitive_deps(rules, target, cache)
        # Remove self
        trans_deps = trans_deps - {target}

        # Classify
        internal_count = 0
        group_flags = {g: 0 for g in EXTERNAL_GROUPS}
        other_repos = set()

        for dep in trans_deps:
            group, is_internal = classify_dep(dep)
            if is_internal:
                internal_count += 1
            elif group in EXTERNAL_GROUPS:
                group_flags[group] = 1
            else:
                # Count unique external repos
                if dep.startswith("@"):
                    repo = dep.split("//")[0]
                    other_repos.add(repo)
                elif dep.startswith("//third_party:"):
                    other_repos.add(dep)
                elif dep.startswith("//:"):
                    other_repos.add(dep)

        other_external = len(other_repos)
        total = (internal_count
                 + sum(group_flags.values())
                 + other_external)

        for src in cpp_srcs:
            # Normalize src label to path
            filepath = src.replace("//", "").replace(":", "/")
            if filepath.startswith("/"):
                filepath = filepath[1:]

            results.append({
                "file": filepath,
                "target": target,
                "internal_deps": internal_count,
                "openvino": group_flags["openvino"],
                "genai": group_flags["genai"],
                "tensorflow": group_flags["tensorflow"],
                "mediapipe": group_flags["mediapipe"],
                "grpc": group_flags["grpc"],
                "protobuf": group_flags["protobuf"],
                "other_external": other_external,
                "total_deps": total,
            })

    # Step 4: Write CSV
    print(f"Step 4: Writing {len(results)} entries to {output}")
    with open(output, "w") as f:
        f.write("file,target,internal_deps,openvino,genai,tensorflow,"
                "mediapipe,grpc,protobuf,other_external,total_deps\n")
        for r in sorted(results, key=lambda x: x["file"]):
            f.write(f"{r['file']},{r['target']},{r['internal_deps']},"
                    f"{r['openvino']},{r['genai']},{r['tensorflow']},"
                    f"{r['mediapipe']},{r['grpc']},{r['protobuf']},"
                    f"{r['other_external']},{r['total_deps']}\n")

    # Summary
    if not results:
        print("\nWARNING: No results generated!")
        return

    total_deps_vals = sorted(r["total_deps"] for r in results)
    int_deps_vals = sorted(r["internal_deps"] for r in results)
    avg_total = sum(total_deps_vals) / len(total_deps_vals)
    avg_int = sum(int_deps_vals) / len(int_deps_vals)
    median_total = total_deps_vals[len(total_deps_vals) // 2]
    max_entry = max(results, key=lambda x: x["total_deps"])

    print()
    print("=== Summary ===")
    print(f"Files:                {len(results)}")
    print(f"Unique targets:       {len(set(r['target'] for r in results))}")
    print(f"Avg total deps/file:  {avg_total:.1f}")
    print(f"Avg internal deps:    {avg_int:.1f}")
    print(f"Median total deps:    {median_total}")
    print(f"Max total deps:       {max_entry['total_deps']}  ({max_entry['file']})")
    print()

    # Distribution
    print("Distribution:")
    buckets = [
        (">200", lambda v: v > 200),
        ("101-200", lambda v: 101 <= v <= 200),
        ("91-100", lambda v: 91 <= v <= 100),
        ("81-90", lambda v: 81 <= v <= 90),
        ("71-80", lambda v: 71 <= v <= 80),
        ("61-70", lambda v: 61 <= v <= 70),
        ("51-60", lambda v: 51 <= v <= 60),
        ("41-50", lambda v: 41 <= v <= 50),
        ("31-40", lambda v: 31 <= v <= 40),
        ("21-30", lambda v: 21 <= v <= 30),
        ("11-20", lambda v: 11 <= v <= 20),
        ("0-10", lambda v: v <= 10),
    ]
    for label, pred in buckets:
        count = sum(1 for v in total_deps_vals if pred(v))
        print(f"  {count:>6} files with {label:>8} deps")

    print()
    print("Top 10 files by dependency count:")
    top = sorted(results, key=lambda x: -x["total_deps"])[:10]
    for r in top:
        f = r["file"]
        if len(f) > 55:
            f = "..." + f[-52:]
        print(f"  {f:<55s} {r['total_deps']:>4} deps  "
              f"(int:{r['internal_deps']} ov:{r['openvino']} ga:{r['genai']} "
              f"tf:{r['tensorflow']} mp:{r['mediapipe']} gr:{r['grpc']} "
              f"pb:{r['protobuf']} ext_other:{r['other_external']})")

    print()
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
