#!/bin/bash
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

# Checks git diff for .md files and reports those where changes
# fall inside fenced code blocks.
# Usage: ./check_md_code_changes.sh <platform> [base_ref]
#   platform: "linux" (checks bash+console) or "windows" (checks bat+console)
#   base_ref: optional git ref to diff against (default: HEAD~1)

set -euo pipefail

PLATFORM="${1:-linux}"
BASE_REF="${2:-HEAD~1}"

case "$PLATFORM" in
    linux)   BLOCK_PATTERN='^```(bash|console)' ; FENCE_LANGS='bash|console' ;;
    windows) BLOCK_PATTERN='^```(bat|console)' ; FENCE_LANGS='bat|console' ;;
    *)       echo "Error: platform must be 'linux' or 'windows'" >&2; exit 1 ;;
esac

# Get list of changed .md files
changed_md_files=$(git diff --name-only "$BASE_REF" HEAD -- '*.md')

if [ -z "$changed_md_files" ]; then
    exit 0
fi

matched_files=()

while IFS= read -r file; do
    [ -z "$file" ] && continue

    # Extract changed line numbers in the new version from hunk headers
    changed_lines=$(git diff --unified=0 "$BASE_REF" HEAD -- "$file" \
        | awk '/^@@/ {
            match($0, /\+([0-9]+)(,([0-9]+))?/, arr)
            start = arr[1]
            count = (arr[3] != "") ? arr[3] : 1
            for (i = start; i < start + count; i++) print i
        }')

    if [ -z "$changed_lines" ]; then
        continue
    fi

    # Check if a code block fence was added, removed, or changed type in the diff
    # (e.g. ```->```bash, ```console->```, or entirely new/deleted fence)
    fence_changed=$(git diff --unified=0 "$BASE_REF" HEAD -- "$file" \
        | grep -qE "^[+-]\`\`\`($FENCE_LANGS)" && echo 1 || true)

    if [ -n "$fence_changed" ]; then
        matched_files+=("$file")
        continue
    fi

    # Use awk to find which lines are inside ```bash/bat/console blocks,
    # then check if any of those overlap with changed lines
    hit=$(awk -v lines="$changed_lines" -v pattern="$BLOCK_PATTERN" '
    BEGIN {
        n = split(lines, arr, "\n")
        for (i = 1; i <= n; i++) changed[arr[i]] = 1
    }
    $0 ~ pattern            { in_block = 1; next }
    /^```/                  { in_block = 0; next }
    in_block && (NR in changed) { print FILENAME; exit }
    ' "$file")

    if [ -n "$hit" ]; then
        matched_files+=("$file")
    fi
done <<< "$changed_md_files"

if [ ${#matched_files[@]} -gt 0 ]; then
    printf '%s\n' "${matched_files[@]}"
fi
