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
Compare before/after build metrics and produce a comprehensive report.

Supports two types of CSV inputs:
  - Dependency metrics  (from measure_deps.sh):   file,target,...,total_deps
  - TU size metrics     (from measure_tu_sizes.py): file,pp_lines,pp_bytes

Usage:
  python3 tools/compare_build_metrics.py \\
      --deps-before deps_before.csv --deps-after deps_after.csv \\
      --tu-before tu_before.csv --tu-after tu_after.csv

  All flags are optional — you can compare deps only, TU only, or both.
"""

import argparse
import csv
import sys
from collections import defaultdict


def load_deps_csv(path):
    """Load deps CSV → {file: {target, internal_deps, ..., total_deps}}.
    When a file appears in multiple targets, keep the entry with the
    MINIMUM non-zero total_deps (smallest real target). Zero-dep entries
    from config variants are ignored unless all entries are zero."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "target": row["target"],
                "internal_deps": int(row["internal_deps"]),
                "openvino": int(row["openvino"]),
                "genai": int(row["genai"]),
                "tensorflow": int(row["tensorflow"]),
                "mediapipe": int(row["mediapipe"]),
                "grpc": int(row["grpc"]),
                "protobuf": int(row["protobuf"]),
                "other_external": int(row["other_external"]),
                "total_deps": int(row["total_deps"]),
            }
            fname = row["file"]
            if fname not in data:
                data[fname] = entry
            else:
                existing = data[fname]["total_deps"]
                new = entry["total_deps"]
                # Prefer non-zero minimum; skip bogus 0-dep entries
                if existing == 0 and new > 0:
                    data[fname] = entry
                elif new > 0 and new < existing:
                    data[fname] = entry
    return data


def load_tu_csv(path):
    """Load TU sizes CSV → {file: {pp_lines, pp_bytes}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["file"]] = {
                "pp_lines": int(row["pp_lines"]),
                "pp_bytes": int(row["pp_bytes"]),
            }
    return data


def load_trans_srcs_csv(path):
    """Load transitive source files CSV → {file: {target, trans_src_count}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "target": row["target"],
                "trans_src_count": int(row["trans_src_count"]),
            }
            fname = row["file"]
            if fname not in data:
                data[fname] = entry
            else:
                # Keep minimum non-zero count (smallest real target)
                existing = data[fname]["trans_src_count"]
                new = entry["trans_src_count"]
                if existing == 0 and new > 0:
                    data[fname] = entry
                elif new > 0 and new < existing:
                    data[fname] = entry
    return data


def load_impact_csv(path):
    """Load change impact CSV → {file: {target, rdep_targets, impacted_srcs}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "target": row["target"],
                "rdep_targets": int(row["rdep_targets"]),
                "impacted_srcs": int(row["impacted_srcs"]),
            }
            fname = row["file"]
            if fname not in data:
                data[fname] = entry
            else:
                # Keep max impact (worst case per file)
                if entry["impacted_srcs"] > data[fname]["impacted_srcs"]:
                    data[fname] = entry
    return data


def load_headers_csv(path):
    """Load header fan-in CSV → {file: {unique_headers, unique_project_headers}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["file"]] = {
                "unique_headers": int(row["unique_headers"]),
                "unique_project_headers": int(row["unique_project_headers"]),
            }
    return data


def percentile(sorted_values, p):
    """Compute the p-th percentile of a sorted list."""
    if not sorted_values:
        return 0
    k = (len(sorted_values) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_values) else f
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def pct_change(before, after):
    """Format percentage change string."""
    if before == 0:
        return "N/A"
    pct = (after - before) / before * 100
    return f"{pct:+.1f}%"


def fmt_count(n):
    return f"{n:,}"


def print_section(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def is_test_file(path):
    return path.startswith("src/test/")


def filter_by_scope(data, scope):
    if data is None:
        return None
    if scope == "all":
        return data
    if scope == "non-test":
        return {k: v for k, v in data.items() if not is_test_file(k)}
    if scope == "test":
        return {k: v for k, v in data.items() if is_test_file(k)}
    raise ValueError(f"Unknown scope: {scope}")


def scope_label(scope):
    if scope == "all":
        return "All Files"
    if scope == "non-test":
        return "Non-Test Files"
    if scope == "test":
        return "Test Files"
    return scope


def format_bucket_label(lower, upper, suffix=""):
    if upper is None:
        return f">={lower}{suffix}"
    return f"{lower}-{upper}{suffix}"


def count_bucket(values, lower, upper):
    if upper is None:
        return sum(1 for value in values if value >= lower)
    return sum(1 for value in values if lower <= value <= upper)


def print_bucket_table(title, before_vals, after_vals, buckets, suffix=""):
    print()
    print(f"  {title}:")
    print(f"  {'Bucket':<24s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"  {'-'*58}")
    for lower, upper in buckets:
        before_count = count_bucket(before_vals, lower, upper)
        after_count = count_bucket(after_vals, lower, upper)
        delta = after_count - before_count
        label = format_bucket_label(lower, upper, suffix)
        print(f"  {label:<24s} {before_count:>10} {after_count:>10} {delta:>+10}")


def compare_deps(before, after, label=""):
    """Compare dependency metrics and print report."""
    title = "METRIC 1: Dependency Fan-in per Source File"
    if label:
        title += f" [{label}]"
    print_section(title)

    all_files = sorted(set(before.keys()) | set(after.keys()))
    common_files = sorted(set(before.keys()) & set(after.keys()))

    # Extract total_deps values
    before_vals = sorted(d["total_deps"] for d in before.values())
    after_vals = sorted(d["total_deps"] for d in after.values())

    before_avg = sum(before_vals) / len(before_vals) if before_vals else 0
    after_avg = sum(after_vals) / len(after_vals) if after_vals else 0

    before_internal = sorted(d["internal_deps"] for d in before.values())
    after_internal = sorted(d["internal_deps"] for d in after.values())
    before_int_avg = sum(before_internal) / len(before_internal) if before_internal else 0
    after_int_avg = sum(after_internal) / len(after_internal) if after_internal else 0

    # --- Headline metrics ---
    print()
    print("  Headline Metrics:")
    print(f"  {'':40s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"  {'-'*72}")
    print(f"  {'Files measured':<40s} {len(before_vals):>10,} {len(after_vals):>10,}")
    print(f"  {'Average total deps/file':<40s} {before_avg:>10.1f} {after_avg:>10.1f} {pct_change(before_avg, after_avg):>10s}")
    print(f"  {'Average internal deps/file':<40s} {before_int_avg:>10.1f} {after_int_avg:>10.1f} {pct_change(before_int_avg, after_int_avg):>10s}")
    print(f"  {'Median total deps':<40s} {percentile(before_vals, 50):>10.0f} {percentile(after_vals, 50):>10.0f}")
    print(f"  {'P90 total deps':<40s} {percentile(before_vals, 90):>10.0f} {percentile(after_vals, 90):>10.0f}")
    print(f"  {'P95 total deps':<40s} {percentile(before_vals, 95):>10.0f} {percentile(after_vals, 95):>10.0f}")
    print(f"  {'Max total deps':<40s} {max(before_vals) if before_vals else 0:>10} {max(after_vals) if after_vals else 0:>10}")

    # --- Bucket distribution ---
    dep_buckets = [
        (0, 10),
        (11, 20),
        (21, 30),
        (31, 40),
        (41, 50),
        (51, 60),
        (61, 70),
        (71, 80),
        (81, 90),
        (91, 100),
        (101, 200),
        (201, None),
    ]
    print_bucket_table(
        "Dependency Distribution Buckets",
        before_vals,
        after_vals,
        dep_buckets,
        suffix=" deps",
    )

    # --- Per-file changes (top improvements) ---
    changes = []
    for f in common_files:
        b = before[f]["total_deps"]
        a = after[f]["total_deps"]
        if b != a:
            changes.append((f, b, a, a - b))

    if changes:
        changes.sort(key=lambda x: x[3])  # Most improved first
        print()
        print("  Top 10 Dependency Reductions:")
        print(f"  {'File':<50s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
        print(f"  {'-'*76}")
        for f, b, a, d in changes[:10]:
            name = f if len(f) <= 50 else "..." + f[-47:]
            print(f"  {name:<50s} {b:>8} {a:>8} {d:>+8}")

        # Worst remaining
        worst = sorted(
            [(f, after[f]["total_deps"]) for f in after],
            key=lambda x: -x[1]
        )[:10]
        print()
        print("  Top 10 Remaining Highest-Dep Files (after):")
        print(f"  {'File':<50s} {'Deps':>8s} {'Target':<30s}")
        print(f"  {'-'*90}")
        for f, d in worst:
            target = after[f]["target"]
            name = f if len(f) <= 50 else "..." + f[-47:]
            tgt = target if len(target) <= 30 else "..." + target[-27:]
            print(f"  {name:<50s} {d:>8} {tgt:<30s}")

    # --- Unique targets analysis ---
    before_targets = set(d["target"] for d in before.values())
    after_targets = set(d["target"] for d in after.values())
    print()
    print(f"  Unique targets containing source files:")
    print(f"    Before: {len(before_targets)}")
    print(f"    After:  {len(after_targets)}")
    if len(after_targets) > len(before_targets):
        print(f"    (+{len(after_targets) - len(before_targets)} new targets = more granular build graph)")


def compare_tu(before, after, label=""):
    """Compare translation unit size metrics and print report."""
    title = "METRIC 2: Translation Unit Sizes (Preprocessed)"
    if label:
        title += f" [{label}]"
    print_section(title)

    common_files = sorted(set(before.keys()) & set(after.keys()))

    before_lines = sorted(d["pp_lines"] for d in before.values())
    after_lines = sorted(d["pp_lines"] for d in after.values())

    before_total = sum(before_lines)
    after_total = sum(after_lines)
    before_avg = before_total / len(before_lines) if before_lines else 0
    after_avg = after_total / len(after_lines) if after_lines else 0

    before_bytes = sum(d["pp_bytes"] for d in before.values())
    after_bytes = sum(d["pp_bytes"] for d in after.values())

    # --- Headline metrics ---
    print()
    print("  Headline Metrics:")
    print(f"  {'':40s} {'Before':>12s} {'After':>12s} {'Change':>10s}")
    print(f"  {'-'*76}")
    print(f"  {'Files measured':<40s} {len(before_lines):>12,} {len(after_lines):>12,}")
    print(f"  {'Total PP lines':<40s} {before_total:>12,} {after_total:>12,} {pct_change(before_total, after_total):>10s}")
    print(f"  {'Total PP size (MB)':<40s} {before_bytes/1024/1024:>12.1f} {after_bytes/1024/1024:>12.1f} {pct_change(before_bytes, after_bytes):>10s}")
    print(f"  {'Average PP lines/file':<40s} {before_avg:>12,.0f} {after_avg:>12,.0f} {pct_change(before_avg, after_avg):>10s}")
    print(f"  {'Median PP lines':<40s} {percentile(before_lines, 50):>12,.0f} {percentile(after_lines, 50):>12,.0f}")
    print(f"  {'P90 PP lines':<40s} {percentile(before_lines, 90):>12,.0f} {percentile(after_lines, 90):>12,.0f}")
    print(f"  {'Max PP lines':<40s} {max(before_lines) if before_lines else 0:>12,} {max(after_lines) if after_lines else 0:>12,}")

    # --- Bucket distribution ---
    pp_buckets = [
        (0, 10000),
        (10001, 25000),
        (25001, 50000),
        (50001, 100000),
        (100001, 200000),
        (200001, 300000),
        (300001, None),
    ]
    print_bucket_table(
        "PP Line Distribution Buckets",
        before_lines,
        after_lines,
        pp_buckets,
        suffix="",
    )

    # --- Per-file changes ---
    changes = []
    for f in common_files:
        b = before[f]["pp_lines"]
        a = after[f]["pp_lines"]
        if b > 0 and abs(a - b) > max(100, b * 0.01):
            changes.append((f, b, a, a - b))

    if changes:
        changes.sort(key=lambda x: x[3])  # Most reduced first
        print()
        print("  Top 10 TU Size Reductions:")
        print(f"  {'File':<50s} {'Before':>10s} {'After':>10s} {'Change':>10s} {'Pct':>8s}")
        print(f"  {'-'*80}")
        for f, b, a, d in changes[:10]:
            name = f if len(f) <= 50 else "..." + f[-47:]
            print(f"  {name:<50s} {b:>10,} {a:>10,} {d:>+10,} {pct_change(b, a):>8s}")

        # Top regressions (if any)
        regressions = [c for c in changes if c[3] > 0]
        if regressions:
            regressions.sort(key=lambda x: -x[3])
            print()
            print("  Top 5 TU Size Regressions (if any):")
            print(f"  {'File':<50s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
            print(f"  {'-'*80}")
            for f, b, a, d in regressions[:5]:
                name = f if len(f) <= 50 else "..." + f[-47:]
                print(f"  {name:<50s} {b:>10,} {a:>10,} {d:>+10,}")

        # Largest remaining TUs
        worst = sorted(after.items(), key=lambda x: -x[1]["pp_lines"])[:10]
        print()
        print("  Top 10 Largest TUs (after):")
        print(f"  {'File':<55s} {'PP Lines':>10s} {'PP KB':>10s}")
        print(f"  {'-'*76}")
        for f, d in worst:
            name = f if len(f) <= 55 else "..." + f[-52:]
            print(f"  {name:<55s} {d['pp_lines']:>10,} {d['pp_bytes']/1024:>10,.0f}")


def compare_trans_srcs(before, after, label=""):
    """Compare transitive source file counts and print report."""
    title = "METRIC 3: Transitive Source Files per TU"
    if label:
        title += f" [{label}]"
    print_section(title)

    common_files = sorted(set(before.keys()) & set(after.keys()))

    before_vals = sorted(d["trans_src_count"] for d in before.values())
    after_vals = sorted(d["trans_src_count"] for d in after.values())

    before_avg = sum(before_vals) / len(before_vals) if before_vals else 0
    after_avg = sum(after_vals) / len(after_vals) if after_vals else 0

    # --- Headline metrics ---
    print()
    print("  Headline Metrics:")
    print(f"  {'':40s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"  {'-'*72}")
    print(f"  {'Files measured':<40s} {len(before_vals):>10,} {len(after_vals):>10,}")
    print(f"  {'Average trans src files/TU':<40s} {before_avg:>10.1f} {after_avg:>10.1f} {pct_change(before_avg, after_avg):>10s}")
    print(f"  {'Median trans src files':<40s} {percentile(before_vals, 50):>10.0f} {percentile(after_vals, 50):>10.0f}")
    print(f"  {'P90 trans src files':<40s} {percentile(before_vals, 90):>10.0f} {percentile(after_vals, 90):>10.0f}")
    print(f"  {'Max trans src files':<40s} {max(before_vals) if before_vals else 0:>10} {max(after_vals) if after_vals else 0:>10}")

    # --- Bucket distribution ---
    trans_buckets = [
        (0, 20),
        (21, 50),
        (51, 100),
        (101, 200),
        (201, 500),
        (501, None),
    ]
    print_bucket_table(
        "Trans Source Files Distribution",
        before_vals,
        after_vals,
        trans_buckets,
        suffix=" srcs",
    )

    # --- Per-file changes ---
    changes = []
    for f in common_files:
        b = before[f]["trans_src_count"]
        a = after[f]["trans_src_count"]
        if b != a:
            changes.append((f, b, a, a - b))

    if changes:
        changes.sort(key=lambda x: x[3])
        print()
        print("  Top 10 Trans Src Reductions:")
        print(f"  {'File':<50s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
        print(f"  {'-'*76}")
        for f, b, a, d in changes[:10]:
            name = f if len(f) <= 50 else "..." + f[-47:]
            print(f"  {name:<50s} {b:>8} {a:>8} {d:>+8}")

    # Largest remaining
    worst = sorted(after.items(), key=lambda x: -x[1]["trans_src_count"])[:10]
    print()
    print("  Top 10 Largest Trans Src Counts (after):")
    print(f"  {'File':<50s} {'Count':>8s} {'Target':<30s}")
    print(f"  {'-'*90}")
    for f, d in worst:
        name = f if len(f) <= 50 else "..." + f[-47:]
        tgt = d["target"]
        tgt = tgt if len(tgt) <= 30 else "..." + tgt[-27:]
        print(f"  {name:<50s} {d['trans_src_count']:>8} {tgt:<30s}")


def compare_impact(before, after, label=""):
    """Compare change impact metrics and print report."""
    title = "METRIC 4: Change Impact (Reverse Deps)"
    if label:
        title += f" [{label}]"
    print_section(title)

    common_files = sorted(set(before.keys()) & set(after.keys()))

    before_vals = sorted(d["impacted_srcs"] for d in before.values())
    after_vals = sorted(d["impacted_srcs"] for d in after.values())

    before_avg = sum(before_vals) / len(before_vals) if before_vals else 0
    after_avg = sum(after_vals) / len(after_vals) if after_vals else 0

    before_rdeps = sorted(d["rdep_targets"] for d in before.values())
    after_rdeps = sorted(d["rdep_targets"] for d in after.values())
    before_rdep_avg = sum(before_rdeps) / len(before_rdeps) if before_rdeps else 0
    after_rdep_avg = sum(after_rdeps) / len(after_rdeps) if after_rdeps else 0

    # --- Headline metrics ---
    print()
    print("  Headline Metrics:")
    print(f"  {'':40s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"  {'-'*72}")
    print(f"  {'Files measured':<40s} {len(before_vals):>10,} {len(after_vals):>10,}")
    print(f"  {'Average impacted srcs/file':<40s} {before_avg:>10.1f} {after_avg:>10.1f} {pct_change(before_avg, after_avg):>10s}")
    print(f"  {'Average rdep targets/file':<40s} {before_rdep_avg:>10.1f} {after_rdep_avg:>10.1f} {pct_change(before_rdep_avg, after_rdep_avg):>10s}")
    print(f"  {'Median impacted srcs':<40s} {percentile(before_vals, 50):>10.0f} {percentile(after_vals, 50):>10.0f}")
    print(f"  {'P90 impacted srcs':<40s} {percentile(before_vals, 90):>10.0f} {percentile(after_vals, 90):>10.0f}")
    print(f"  {'Max impacted srcs':<40s} {max(before_vals) if before_vals else 0:>10} {max(after_vals) if after_vals else 0:>10}")

    # --- Bucket distribution ---
    impact_buckets = [
        (0, 50),
        (51, 100),
        (101, 200),
        (201, 500),
        (501, None),
    ]
    print_bucket_table(
        "Change Impact Distribution",
        before_vals,
        after_vals,
        impact_buckets,
        suffix=" srcs",
    )

    # --- Per-file changes ---
    changes = []
    for f in common_files:
        b = before[f]["impacted_srcs"]
        a = after[f]["impacted_srcs"]
        if b != a:
            changes.append((f, b, a, a - b))

    if changes:
        changes.sort(key=lambda x: x[3])
        print()
        print("  Top 10 Impact Reductions:")
        print(f"  {'File':<50s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
        print(f"  {'-'*76}")
        for f, b, a, d in changes[:10]:
            name = f if len(f) <= 50 else "..." + f[-47:]
            print(f"  {name:<50s} {b:>8} {a:>8} {d:>+8}")

    # Largest remaining
    worst = sorted(after.items(), key=lambda x: -x[1]["impacted_srcs"])[:10]
    print()
    print("  Top 10 Largest Change Impact (after):")
    print(f"  {'File':<50s} {'Impact':>8s} {'Rdeps':>8s} {'Target':<25s}")
    print(f"  {'-'*93}")
    for f, d in worst:
        name = f if len(f) <= 50 else "..." + f[-47:]
        tgt = d["target"]
        tgt = tgt if len(tgt) <= 25 else "..." + tgt[-22:]
        print(f"  {name:<50s} {d['impacted_srcs']:>8} {d['rdep_targets']:>8} {tgt:<25s}")


def compare_headers(before, after, label=""):
    """Compare header fan-in metrics and print report."""
    title = "METRIC 5: Header Fan-in per TU"
    if label:
        title += f" [{label}]"
    print_section(title)

    common_files = sorted(set(before.keys()) & set(after.keys()))

    before_vals = sorted(d["unique_headers"] for d in before.values())
    after_vals = sorted(d["unique_headers"] for d in after.values())

    before_avg = sum(before_vals) / len(before_vals) if before_vals else 0
    after_avg = sum(after_vals) / len(after_vals) if after_vals else 0

    before_proj = sorted(d["unique_project_headers"] for d in before.values())
    after_proj = sorted(d["unique_project_headers"] for d in after.values())
    before_proj_avg = sum(before_proj) / len(before_proj) if before_proj else 0
    after_proj_avg = sum(after_proj) / len(after_proj) if after_proj else 0

    # --- Headline metrics ---
    print()
    print("  Headline Metrics:")
    print(f"  {'':40s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"  {'-'*72}")
    print(f"  {'Files measured':<40s} {len(before_vals):>10,} {len(after_vals):>10,}")
    print(f"  {'Average unique headers/file':<40s} {before_avg:>10.1f} {after_avg:>10.1f} {pct_change(before_avg, after_avg):>10s}")
    print(f"  {'Average project headers/file':<40s} {before_proj_avg:>10.1f} {after_proj_avg:>10.1f} {pct_change(before_proj_avg, after_proj_avg):>10s}")
    print(f"  {'Median unique headers':<40s} {percentile(before_vals, 50):>10.0f} {percentile(after_vals, 50):>10.0f}")
    print(f"  {'P90 unique headers':<40s} {percentile(before_vals, 90):>10.0f} {percentile(after_vals, 90):>10.0f}")
    print(f"  {'Max unique headers':<40s} {max(before_vals) if before_vals else 0:>10} {max(after_vals) if after_vals else 0:>10}")

    # --- Bucket distribution ---
    header_buckets = [
        (0, 100),
        (101, 200),
        (201, 500),
        (501, 1000),
        (1001, 2000),
        (2001, None),
    ]
    print_bucket_table(
        "Unique Headers Distribution",
        before_vals,
        after_vals,
        header_buckets,
        suffix="",
    )

    # --- Per-file changes ---
    changes = []
    for f in common_files:
        b = before[f]["unique_headers"]
        a = after[f]["unique_headers"]
        if b != a and abs(a - b) > max(5, b * 0.02):
            changes.append((f, b, a, a - b))

    if changes:
        changes.sort(key=lambda x: x[3])
        print()
        print("  Top 10 Header Count Reductions:")
        print(f"  {'File':<50s} {'Before':>8s} {'After':>8s} {'Change':>8s} {'Pct':>8s}")
        print(f"  {'-'*78}")
        for f, b, a, d in changes[:10]:
            name = f if len(f) <= 50 else "..." + f[-47:]
            print(f"  {name:<50s} {b:>8} {a:>8} {d:>+8} {pct_change(b, a):>8s}")

    # Largest remaining
    worst = sorted(after.items(), key=lambda x: -x[1]["unique_headers"])[:10]
    print()
    print("  Top 10 Highest Header Fan-in (after):")
    print(f"  {'File':<55s} {'Total':>8s} {'Project':>8s}")
    print(f"  {'-'*73}")
    for f, d in worst:
        name = f if len(f) <= 55 else "..." + f[-52:]
        print(f"  {name:<55s} {d['unique_headers']:>8} {d['unique_project_headers']:>8}")


def print_verdict(deps_before, deps_after, tu_before, tu_after, label="",
                  trans_before=None, trans_after=None,
                  impact_before=None, impact_after=None,
                  headers_before=None, headers_after=None):
    """Print overall verdict."""
    title = "VERDICT"
    if label:
        title += f" [{label}]"
    print_section(title)
    print()

    verdicts = []

    if deps_before and deps_after:
        b_avg = sum(d["total_deps"] for d in deps_before.values()) / len(deps_before)
        a_avg = sum(d["total_deps"] for d in deps_after.values()) / len(deps_after)
        if a_avg < b_avg:
            pct = (b_avg - a_avg) / b_avg * 100
            verdicts.append(f"  [IMPROVED] Dependency fan-in reduced by {pct:.1f}% on average")
        elif a_avg > b_avg:
            pct = (a_avg - b_avg) / b_avg * 100
            verdicts.append(f"  [REGRESSED] Dependency fan-in increased by {pct:.1f}% on average")
        else:
            verdicts.append(f"  [UNCHANGED] Dependency fan-in unchanged")

        # Count files above high threshold
        b_high = sum(1 for d in deps_before.values() if d["total_deps"] > 100)
        a_high = sum(1 for d in deps_after.values() if d["total_deps"] > 100)
        if a_high < b_high:
            verdicts.append(f"  [IMPROVED] High-dep files (>100): {b_high} -> {a_high} (-{b_high - a_high})")

    if tu_before and tu_after:
        b_total = sum(d["pp_lines"] for d in tu_before.values())
        a_total = sum(d["pp_lines"] for d in tu_after.values())
        if a_total < b_total:
            pct = (b_total - a_total) / b_total * 100
            verdicts.append(f"  [IMPROVED] Total preprocessed lines reduced by {pct:.1f}%")
        elif a_total > b_total:
            pct = (a_total - b_total) / b_total * 100
            verdicts.append(f"  [REGRESSED] Total preprocessed lines increased by {pct:.1f}%")
        else:
            verdicts.append(f"  [UNCHANGED] Total preprocessed lines unchanged")

    if trans_before and trans_after:
        b_avg = sum(d["trans_src_count"] for d in trans_before.values()) / len(trans_before)
        a_avg = sum(d["trans_src_count"] for d in trans_after.values()) / len(trans_after)
        if a_avg < b_avg:
            pct = (b_avg - a_avg) / b_avg * 100
            verdicts.append(f"  [IMPROVED] Trans source files reduced by {pct:.1f}% on average")
        elif a_avg > b_avg:
            pct = (a_avg - b_avg) / b_avg * 100
            verdicts.append(f"  [REGRESSED] Trans source files increased by {pct:.1f}% on average")

    if impact_before and impact_after:
        b_avg = sum(d["impacted_srcs"] for d in impact_before.values()) / len(impact_before)
        a_avg = sum(d["impacted_srcs"] for d in impact_after.values()) / len(impact_after)
        if a_avg < b_avg:
            pct = (b_avg - a_avg) / b_avg * 100
            verdicts.append(f"  [IMPROVED] Change impact reduced by {pct:.1f}% on average")
        elif a_avg > b_avg:
            pct = (a_avg - b_avg) / b_avg * 100
            verdicts.append(f"  [REGRESSED] Change impact increased by {pct:.1f}% on average")

    if headers_before and headers_after:
        b_avg = sum(d["unique_headers"] for d in headers_before.values()) / len(headers_before)
        a_avg = sum(d["unique_headers"] for d in headers_after.values()) / len(headers_after)
        if a_avg < b_avg:
            pct = (b_avg - a_avg) / b_avg * 100
            verdicts.append(f"  [IMPROVED] Header fan-in reduced by {pct:.1f}% on average")
        elif a_avg > b_avg:
            pct = (a_avg - b_avg) / b_avg * 100
            verdicts.append(f"  [REGRESSED] Header fan-in increased by {pct:.1f}% on average")

    for v in verdicts:
        print(v)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare before/after build metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare deps only:
  %(prog)s --deps-before deps_v1.csv --deps-after deps_v2.csv

  # Compare TU sizes only:
  %(prog)s --tu-before tu_v1.csv --tu-after tu_v2.csv

  # Compare both:
  %(prog)s --deps-before deps_v1.csv --deps-after deps_v2.csv \\
           --tu-before tu_v1.csv --tu-after tu_v2.csv

  # Compare all 5 metrics:
  %(prog)s --deps-before deps_v1.csv --deps-after deps_v2.csv \\
           --tu-before tu_v1.csv --tu-after tu_v2.csv \\
           --trans-srcs-before trans_v1.csv --trans-srcs-after trans_v2.csv \\
           --impact-before impact_v1.csv --impact-after impact_v2.csv \\
           --headers-before headers_v1.csv --headers-after headers_v2.csv
"""
    )
    parser.add_argument("--before-dir",
                        help="Directory with before CSVs (auto-resolves deps.csv, tu_src.csv, etc.)")
    parser.add_argument("--after-dir",
                        help="Directory with after CSVs (auto-resolves deps.csv, tu_src.csv, etc.)")
    parser.add_argument("--deps-before", help="Deps CSV from before changes")
    parser.add_argument("--deps-after", help="Deps CSV from after changes")
    parser.add_argument("--tu-before", help="TU sizes CSV from before changes")
    parser.add_argument("--tu-after", help="TU sizes CSV from after changes")
    parser.add_argument("--trans-srcs-before", help="Trans source files CSV from before")
    parser.add_argument("--trans-srcs-after", help="Trans source files CSV from after")
    parser.add_argument("--impact-before", help="Change impact CSV from before")
    parser.add_argument("--impact-after", help="Change impact CSV from after")
    parser.add_argument("--headers-before", help="Header fan-in CSV from before")
    parser.add_argument("--headers-after", help="Header fan-in CSV from after")
    parser.add_argument(
        "--scopes",
        default="all",
        help=(
            "Comma-separated file scopes to report: all,non-test,test. "
            "Example: --scopes all,non-test,test"
        ),
    )
    args = parser.parse_args()

    # Resolve --before-dir / --after-dir to individual CSV paths
    import os
    DIR_FILE_MAP = {
        "deps":       ("deps_before",       "deps_after",       "deps.csv"),
        "tu":         ("tu_before",          "tu_after",         "tu_src.csv"),
        "trans_srcs": ("trans_srcs_before",  "trans_srcs_after", "trans_srcs.csv"),
        "impact":     ("impact_before",      "impact_after",     "impact.csv"),
        "headers":    ("headers_before",     "headers_after",    "headers_src.csv"),
    }
    for _metric, (before_attr, after_attr, filename) in DIR_FILE_MAP.items():
        if args.before_dir and not getattr(args, before_attr):
            candidate = os.path.join(args.before_dir, filename)
            if os.path.isfile(candidate):
                setattr(args, before_attr, candidate)
        if args.after_dir and not getattr(args, after_attr):
            candidate = os.path.join(args.after_dir, filename)
            if os.path.isfile(candidate):
                setattr(args, after_attr, candidate)

    if not any([args.deps_before, args.tu_before, args.trans_srcs_before,
                args.impact_before, args.headers_before]):
        parser.error("At least one --*-before / --*-after pair is required.")

    deps_before = deps_after = tu_before = tu_after = None
    trans_before = trans_after = impact_before = impact_after = None
    headers_before = headers_after = None

    if args.deps_before and args.deps_after:
        deps_before = load_deps_csv(args.deps_before)
        deps_after = load_deps_csv(args.deps_after)
    elif args.deps_before or args.deps_after:
        print("WARNING: Need both --deps-before and --deps-after for comparison.",
              file=sys.stderr)

    if args.tu_before and args.tu_after:
        tu_before = load_tu_csv(args.tu_before)
        tu_after = load_tu_csv(args.tu_after)
    elif args.tu_before or args.tu_after:
        print("WARNING: Need both --tu-before and --tu-after for comparison.",
              file=sys.stderr)

    if args.trans_srcs_before and args.trans_srcs_after:
        trans_before = load_trans_srcs_csv(args.trans_srcs_before)
        trans_after = load_trans_srcs_csv(args.trans_srcs_after)
    elif args.trans_srcs_before or args.trans_srcs_after:
        print("WARNING: Need both --trans-srcs-before and --trans-srcs-after.",
              file=sys.stderr)

    if args.impact_before and args.impact_after:
        impact_before = load_impact_csv(args.impact_before)
        impact_after = load_impact_csv(args.impact_after)
    elif args.impact_before or args.impact_after:
        print("WARNING: Need both --impact-before and --impact-after.",
              file=sys.stderr)

    if args.headers_before and args.headers_after:
        headers_before = load_headers_csv(args.headers_before)
        headers_after = load_headers_csv(args.headers_after)
    elif args.headers_before or args.headers_after:
        print("WARNING: Need both --headers-before and --headers-after.",
              file=sys.stderr)

    requested_scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    valid_scopes = {"all", "non-test", "test"}
    invalid = [s for s in requested_scopes if s not in valid_scopes]
    if invalid:
        parser.error(f"Invalid scope(s): {', '.join(invalid)}. Valid scopes: all,non-test,test")

    # Remove duplicates while preserving order
    seen = set()
    scopes = []
    for s in requested_scopes:
        if s not in seen:
            scopes.append(s)
            seen.add(s)

    for idx, scope in enumerate(scopes):
        if idx > 0:
            print()

        deps_b = filter_by_scope(deps_before, scope) if deps_before is not None else None
        deps_a = filter_by_scope(deps_after, scope) if deps_after is not None else None
        tu_b = filter_by_scope(tu_before, scope) if tu_before is not None else None
        tu_a = filter_by_scope(tu_after, scope) if tu_after is not None else None
        trans_b = filter_by_scope(trans_before, scope) if trans_before is not None else None
        trans_a = filter_by_scope(trans_after, scope) if trans_after is not None else None
        impact_b = filter_by_scope(impact_before, scope) if impact_before is not None else None
        impact_a = filter_by_scope(impact_after, scope) if impact_after is not None else None
        headers_b = filter_by_scope(headers_before, scope) if headers_before is not None else None
        headers_a = filter_by_scope(headers_after, scope) if headers_after is not None else None

        label = scope_label(scope)
        printed = False

        if deps_b is not None and deps_a is not None:
            if deps_b and deps_a:
                compare_deps(deps_b, deps_a, label)
                printed = True
            else:
                print_section(f"METRIC 1: Dependency Fan-in per Source File [{label}]")
                print("\n  No dependency rows in this scope.")
                printed = True

        if tu_b is not None and tu_a is not None:
            if tu_b and tu_a:
                compare_tu(tu_b, tu_a, label)
                printed = True
            else:
                print_section(f"METRIC 2: Translation Unit Sizes (Preprocessed) [{label}]")
                print("\n  No TU rows in this scope.")
                printed = True

        if trans_b is not None and trans_a is not None:
            if trans_b and trans_a:
                compare_trans_srcs(trans_b, trans_a, label)
                printed = True
            else:
                print_section(f"METRIC 3: Transitive Source Files per TU [{label}]")
                print("\n  No trans source files rows in this scope.")
                printed = True

        if impact_b is not None and impact_a is not None:
            if impact_b and impact_a:
                compare_impact(impact_b, impact_a, label)
                printed = True
            else:
                print_section(f"METRIC 4: Change Impact (Reverse Deps) [{label}]")
                print("\n  No change impact rows in this scope.")
                printed = True

        if headers_b is not None and headers_a is not None:
            if headers_b and headers_a:
                compare_headers(headers_b, headers_a, label)
                printed = True
            else:
                print_section(f"METRIC 5: Header Fan-in per TU [{label}]")
                print("\n  No header fan-in rows in this scope.")
                printed = True

        if printed:
            print_verdict(deps_b, deps_a, tu_b, tu_a, label,
                          trans_before=trans_b, trans_after=trans_a,
                          impact_before=impact_b, impact_after=impact_a,
                          headers_before=headers_b, headers_after=headers_a)


if __name__ == "__main__":
    main()
