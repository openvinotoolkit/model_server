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
"""Rewrite links pinned to a moving target so they point at a specific release instead.

When a release branch is cut (e.g. releases/2026/4), two kinds of links in docs/demos
need to be pinned to that release instead of tracking the latest content:

1. Links to this repository's default branch:
   github.com/openvinotoolkit/model_server/blob|tree/main/... or
   raw.githubusercontent.com/openvinotoolkit/model_server/releases/2026/4/...
   are rewritten to use the release branch instead of "main". Only links to this
   repository (openvinotoolkit/model_server) are rewritten. Links to other OpenVINO
   components (openvino, openvino.genai, openvino_tokenizers) or any other repository
   are left untouched, since their release branches are not tied to this repo's release
   branch.

2. Links to the nightly OpenVINO docs site:
   docs.openvino.ai/2026/... is rewritten to docs.openvino.ai/<docs-version>/...
   (e.g. 2026), so readers of a release branch land on docs matching that release
   instead of the ever-changing nightly build. By default the docs version is derived
   from the release branch (releases/2026/4 -> 2026); pass --docs-version to override it.

Note: build-time version variables (e.g. versions.mk package URLs) are intentionally
not touched by this script, since changing them affects what gets built/packaged and
should be done as a separate, deliberate change.

Pass --check-links to additionally fetch every link that now points at the release
branch / docs version (whether rewritten in this run or already correct from before)
and verify each one responds with HTTP 200. This also works with --dry-run, since the
check runs against the in-memory rewritten content, not against what's on disk.

Usage:
    python ci/update_release_links.py --release-branch releases/2026/4
    python ci/update_release_links.py --release-branch releases/2026/4 --docs-version 2026
    python ci/update_release_links.py --release-branch releases/2026/4 --dry-run
    python ci/update_release_links.py --release-branch releases/2026/4 --check-links
"""
import re
import sys
import subprocess
import argparse
import urllib.error
import urllib.request
import concurrent.futures

REPO_PATH = "openvinotoolkit/model_server"
DEFAULT_BRANCH = "main"
DOCS_NIGHTLY_VERSION = "nightly"
# Trailing characters that commonly terminate a URL in markdown/text but aren't part of it.
URL_TAIL = r"[^\s)\]\\\"'<>]+"


def build_repo_link_patterns() -> list:
    repo_re = re.escape(REPO_PATH)
    branch_re = re.escape(DEFAULT_BRANCH)
    return [
        re.compile(rf"(github\.com/{repo_re}/(?:blob|tree)/){branch_re}"
                   rf"(/|(?=[?#\s)\]\\\"'<>]|$))"),
        re.compile(rf"(raw\.githubusercontent\.com/{repo_re}/){branch_re}(/)"),
        re.compile(rf"(raw\.githubusercontent\.com/{repo_re}/refs/heads/){branch_re}(/)"),
    ]


def build_docs_link_pattern():
    return re.compile(rf"(docs\.openvino\.ai/){DOCS_NIGHTLY_VERSION}(/)")


def derive_docs_version(release_branch: str):
    match = re.match(r"releases/(\d{4})(?:/\d+)?$", release_branch)
    return match.group(1) if match else None


def find_release_urls(content: str, release_branch: str, docs_version) -> list:
    repo_re = re.escape(REPO_PATH)
    branch_re = re.escape(release_branch)
    url_patterns = [
        rf"https?://github\.com/{repo_re}/(?:blob|tree)/{branch_re}"
        rf"(?:/{URL_TAIL}|[?#]{URL_TAIL})?(?=[\s)\]\\\"'<>]|$)",
        rf"https?://raw\.githubusercontent\.com/{repo_re}/(?:refs/heads/)?{branch_re}/{URL_TAIL}",
    ]
    if docs_version:
        url_patterns.append(rf"https?://docs\.openvino\.ai/{re.escape(docs_version)}/{URL_TAIL}")
    urls = []
    for pattern in url_patterns:
        urls.extend(re.findall(pattern, content))
    return urls


def check_url(url: str, timeout: int):
    headers = {"User-Agent": "ovms-release-link-checker"}
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD", headers=headers), timeout=timeout) as response:
            if response.status == 200:
                return response.status
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass
    # HEAD can be unsupported or fail transiently even when GET succeeds.
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except urllib.error.URLError:
        return None


def check_links(links: dict, timeout: int, jobs: int) -> int:
    print(f"\nChecking {len(links)} link(s)...")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_url = {executor.submit(check_url, url, timeout): url for url in links}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            status = future.result()
            if status != 200:
                failures.append((links[url], url, status))

    if not failures:
        print(f"All {len(links)} link(s) returned 200.")
        return 0

    print(f"{len(failures)} link(s) did not return 200:")
    for path, url, status in sorted(failures):
        print(f"  [{status if status is not None else 'ERROR'}] {url}  (first seen in {path})")
    return 1


def tracked_files() -> list:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [line for line in output.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-branch", required=True,
                         help="Target branch/tag to point links to, e.g. releases/2026/4")
    parser.add_argument("--docs-version", default=None,
                         help="docs.openvino.ai version to replace 'nightly' with "
                         "(default: derived from --release-branch, e.g. 2026)")
    parser.add_argument("--dry-run", action="store_true",
                         help="Only list files that would change, without writing them")
    parser.add_argument("--check-links", action="store_true",
                         help="Fetch every link now pointing at the release branch/docs version "
                         "and verify it returns HTTP 200")
    parser.add_argument("--timeout", type=int, default=10,
                         help="Per-request timeout in seconds for --check-links (default: 10)")
    parser.add_argument("--jobs", type=int, default=8,
                         help="Concurrent requests for --check-links (default: 8)")
    args = parser.parse_args()

    repo_link_patterns = build_repo_link_patterns()

    docs_version = args.docs_version or derive_docs_version(args.release_branch)
    docs_link_pattern = build_docs_link_pattern() if docs_version else None
    if not docs_version:
        print("Note: could not derive a docs.openvino.ai version from --release-branch; "
              "pass --docs-version to also rewrite 'docs.openvino.ai/nightly' links.")

    changed_files = 0
    found_links = {}  # url -> first file it was seen in
    for path in tracked_files():
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue

        new_content = content
        for pattern in repo_link_patterns:
            new_content = pattern.sub(rf"\g<1>{args.release_branch}\g<2>", new_content)
        if docs_link_pattern:
            new_content = docs_link_pattern.sub(rf"\g<1>{docs_version}\g<2>", new_content)

        if args.check_links:
            for url in find_release_urls(new_content, args.release_branch, docs_version):
                found_links.setdefault(url, path)

        if new_content == content:
            continue

        changed_files += 1
        if args.dry_run:
            print(f"Would update: {path}")
        else:
            with open(path, "w", encoding="utf-8") as file:
                file.write(new_content)
            print(f"Updated: {path}")

    verb = "Would update" if args.dry_run else "Updated"
    print(f"{verb} {changed_files} file(s).")

    if args.check_links:
        exit_code = check_links(found_links, timeout=args.timeout, jobs=args.jobs)
        if exit_code != 0:
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
