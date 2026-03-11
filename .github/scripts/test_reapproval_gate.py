#!/usr/bin/env python3
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the reapproval_gate.py core business logic.

These tests exercise only pure functions that do **not** call the GitHub API,
so they run offline with no credentials required.

Run with:
    python .github/scripts/test_reapproval_gate.py
"""

import os
import sys
import unittest

# Ensure the scripts directory is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reapproval_gate import (  # noqa: E402  (import after sys.path manipulation)
    LOC_THRESHOLD,
    find_latest_approval,
    is_merge_from_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_commit(message: str, num_parents: int = 1) -> dict:
    """Build a minimal commit payload as returned by the GitHub API."""
    return {
        "commit": {"message": message},
        "parents": [{"sha": f"parent{i}"} for i in range(num_parents)],
    }


def _make_review(state: str, login: str = "reviewer", commit_id: str = "abc123") -> dict:
    """Build a minimal review payload as returned by the GitHub API."""
    return {"state": state, "user": {"login": login}, "commit_id": commit_id}


# ---------------------------------------------------------------------------
# is_merge_from_main
# ---------------------------------------------------------------------------

class TestIsMergeFromMain(unittest.TestCase):
    """Tests for is_merge_from_main()."""

    def test_regular_commit_single_parent(self):
        """A normal commit with one parent is never a merge."""
        commit = _make_commit("Fix a bug", num_parents=1)
        self.assertFalse(is_merge_from_main(commit))

    def test_merge_from_main_standard_message(self):
        """Standard 'Merge branch 'main'' message with two parents."""
        commit = _make_commit("Merge branch 'main' into feature-xyz", num_parents=2)
        self.assertTrue(is_merge_from_main(commit))

    def test_merge_from_main_without_into_clause(self):
        """Short-form merge message created by some git flows."""
        commit = _make_commit("Merge branch 'main'", num_parents=2)
        self.assertTrue(is_merge_from_main(commit))

    def test_merge_from_main_origin_remote_tracking(self):
        """GitHub 'Update branch' button uses this message format."""
        commit = _make_commit(
            "Merge remote-tracking branch 'origin/main' into feature",
            num_parents=2,
        )
        self.assertTrue(is_merge_from_main(commit))

    def test_merge_from_main_upstream_remote_tracking(self):
        """Upstream remote tracking branch variant."""
        commit = _make_commit(
            "Merge remote-tracking branch 'upstream/main'",
            num_parents=2,
        )
        self.assertTrue(is_merge_from_main(commit))

    def test_merge_from_main_refs_heads(self):
        """refs/heads/ variant occasionally produced by tooling."""
        commit = _make_commit("Merge refs/heads/main", num_parents=2)
        self.assertTrue(is_merge_from_main(commit))

    def test_merge_from_other_branch_not_main(self):
        """Merging a non-main branch should not match."""
        commit = _make_commit("Merge branch 'feature-other' into feature-xyz", num_parents=2)
        self.assertFalse(is_merge_from_main(commit))

    def test_no_parents_not_merge(self):
        """A root commit (no parents) cannot be a merge."""
        commit = _make_commit("Initial commit", num_parents=0)
        self.assertFalse(is_merge_from_main(commit))

    def test_single_parent_with_merge_message_not_merge(self):
        """Message alone is not sufficient; two parents are required."""
        commit = _make_commit("Merge branch 'main' into feature", num_parents=1)
        self.assertFalse(is_merge_from_main(commit))

    def test_case_insensitive_matching(self):
        """Pattern matching must be case-insensitive."""
        commit = _make_commit("Merge Branch 'main' Into Feature", num_parents=2)
        self.assertTrue(is_merge_from_main(commit))

    def test_custom_main_branch_name(self):
        """The main_branch parameter allows different default-branch names."""
        commit = _make_commit("Merge branch 'develop'", num_parents=2)
        self.assertTrue(is_merge_from_main(commit, main_branch="develop"))

    def test_custom_main_branch_does_not_match_wrong_branch(self):
        """Custom branch name must not accidentally match other branches."""
        commit = _make_commit("Merge branch 'develop'", num_parents=2)
        self.assertFalse(is_merge_from_main(commit, main_branch="main"))

    def test_special_regex_chars_in_branch_name_escaped(self):
        """Branch names with regex special characters are handled safely."""
        commit = _make_commit("Merge branch 'release/1.0'", num_parents=2)
        self.assertTrue(is_merge_from_main(commit, main_branch="release/1.0"))

    def test_three_parent_merge_from_main(self):
        """Octopus merges (3+ parents) with main-branch message are detected."""
        commit = _make_commit("Merge branch 'main' into feature", num_parents=3)
        self.assertTrue(is_merge_from_main(commit))

    def test_squash_commit_with_main_mention_not_merge(self):
        """A squash commit (1 parent) mentioning 'main' is not a merge."""
        commit = _make_commit(
            "Squash and merge: Merge branch 'main' fixes",
            num_parents=1,
        )
        self.assertFalse(is_merge_from_main(commit))


# ---------------------------------------------------------------------------
# find_latest_approval
# ---------------------------------------------------------------------------

class TestFindLatestApproval(unittest.TestCase):
    """Tests for find_latest_approval()."""

    def test_empty_reviews_returns_none(self):
        self.assertIsNone(find_latest_approval([]))

    def test_single_approval_returned(self):
        reviews = [_make_review("APPROVED", commit_id="sha1")]
        result = find_latest_approval(reviews)
        self.assertIsNotNone(result)
        self.assertEqual(result["commit_id"], "sha1")

    def test_only_changes_requested_returns_none(self):
        reviews = [_make_review("CHANGES_REQUESTED")]
        self.assertIsNone(find_latest_approval(reviews))

    def test_only_comment_returns_none(self):
        reviews = [_make_review("COMMENTED")]
        self.assertIsNone(find_latest_approval(reviews))

    def test_multiple_approvals_latest_returned(self):
        """The last APPROVED review in chronological order is returned."""
        reviews = [
            _make_review("APPROVED", commit_id="sha1"),
            _make_review("APPROVED", commit_id="sha2"),
        ]
        result = find_latest_approval(reviews)
        self.assertEqual(result["commit_id"], "sha2")

    def test_approval_after_changes_requested(self):
        """A fresh approval after CHANGES_REQUESTED clears the block."""
        reviews = [
            _make_review("CHANGES_REQUESTED", commit_id="sha1"),
            _make_review("APPROVED", commit_id="sha2"),
        ]
        result = find_latest_approval(reviews)
        self.assertEqual(result["commit_id"], "sha2")

    def test_changes_requested_after_approval(self):
        """If CHANGES_REQUESTED follows an approval, the approval is still tracked."""
        reviews = [
            _make_review("APPROVED", commit_id="sha1"),
            _make_review("CHANGES_REQUESTED", commit_id="sha2"),
        ]
        result = find_latest_approval(reviews)
        # The most recent APPROVED review is sha1
        self.assertEqual(result["commit_id"], "sha1")

    def test_commented_reviews_ignored(self):
        """COMMENTED reviews have no effect on the latest approval."""
        reviews = [
            _make_review("COMMENTED"),
            _make_review("APPROVED", commit_id="sha1"),
            _make_review("COMMENTED"),
        ]
        result = find_latest_approval(reviews)
        self.assertEqual(result["commit_id"], "sha1")

    def test_dismissed_review_not_approval(self):
        """DISMISSED reviews are not counted as approvals."""
        reviews = [_make_review("DISMISSED", commit_id="sha1")]
        self.assertIsNone(find_latest_approval(reviews))

    def test_multiple_reviewers_latest_approval_wins(self):
        """Latest approval wins even if it comes from a different reviewer."""
        reviews = [
            _make_review("APPROVED", login="alice", commit_id="sha1"),
            _make_review("APPROVED", login="bob", commit_id="sha2"),
        ]
        result = find_latest_approval(reviews)
        self.assertEqual(result["commit_id"], "sha2")
        self.assertEqual(result["user"]["login"], "bob")


# ---------------------------------------------------------------------------
# LOC_THRESHOLD constant
# ---------------------------------------------------------------------------

class TestLOCThreshold(unittest.TestCase):
    """Sanity-check the published threshold value."""

    def test_threshold_is_ten(self):
        self.assertEqual(LOC_THRESHOLD, 10)

    def test_threshold_is_int(self):
        self.assertIsInstance(LOC_THRESHOLD, int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    sys.exit(0 if result.wasSuccessful() else 1)
