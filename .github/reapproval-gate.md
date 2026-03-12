# Reapproval Gate

The reapproval gate is an automated GitHub Actions workflow that prevents
approved pull requests from being silently merged after significant new code is
introduced without a fresh review.

## How it works

The gate runs whenever **new commits are pushed to an open PR** (`synchronize`
event), or whenever a **review is submitted** (so the gate re-evaluates after a
fresh approval).

Each reviewer's approval is evaluated **independently**:

```
             For each reviewer who
             has approved the PR:
                      │
       ───────────────┼──────────────────
       ▼                                ▼
  Approval is the              Commits exist after
  most recent commit?          reviewer's approval?
       │                                │
       ▼                        No      │      Yes
     VALID                 ─────────────┼──────────────
                           ▼                           ▼
                         VALID           Calculate changes*
                                                │
                                        changes > 20 LOC?
                                                │
                                        No      │      Yes
                                   ─────────────┼──────────
                                   ▼                       ▼
                                 VALID                   STALE
                                                  (re-request that
                                                    reviewer only)
```

Gate result:
- **PASS** — every approver's approval is VALID.
- **FAIL** — at least one approver is STALE (re-review is re-requested only from
  stale approvers; reviewers with a current approval are not bothered).

\* "Changes" counts **additions + deletions** across commits pushed after the
reviewer's most recent approval, **excluding** merge-from-main commits (e.g.
the commits created by GitHub's "Update branch" button or `git merge main`).
Merge-from-main commits are skipped because they only bring in code that was
already reviewed and merged into the main branch — they represent no new work
by the PR author.

## Required check

The job is named **`reapproval-gate`** (the full GitHub check name visible in
branch protection settings is `Reapproval Gate / reapproval-gate`).

To enforce the gate, add it as a **required status check** in your branch
protection rule:

1. Go to **Settings → Branches → Branch protection rules** for the `main`
   branch.
2. Enable **"Require status checks to pass before merging"**.
3. Search for and add **`Reapproval Gate / reapproval-gate`** (or just
   `reapproval-gate` depending on how GitHub resolves the name in your
   settings).
4. Optionally enable **"Require branches to be up to date before merging"**
   for additional safety.

## Configuration

All configuration lives in the workflow file
`.github/workflows/reapproval-gate.yml`.

| Variable | Default | Description |
|---|---|---|
| `MAIN_BRANCH` | `main` | Name of the default branch. Merge-from-`MAIN_BRANCH` commits are excluded from the changes calculation. |
| `LOC_THRESHOLD` | `20` | Maximum allowed additions + deletions after a reviewer's most recent approval before re-approval is required from that reviewer. Change this constant in `.github/scripts/reapproval_gate.py`. |

## Merge-from-main detection

A commit is treated as a "merge from main" (and skipped in the LOC count) when
**both** of the following are true:

1. The commit has **two or more parents** (it is a merge commit).
2. The commit message matches one of these patterns (case-insensitive):
   - `Merge branch 'main'`
   - `Merge remote-tracking branch 'origin/main'`
   - `Merge remote-tracking branch 'upstream/main'`
   - `Merge refs/heads/main`

These patterns match the messages produced by `git merge main` and by
GitHub's **"Update branch"** button.  Custom merge messages that don't follow
these patterns will be treated as regular commits and counted toward the LOC
threshold.

## What happens when the gate fails

1. The `reapproval-gate` check turns red and blocks merge (when required).
2. The workflow **re-requests review only from stale approvers** — reviewers
   whose approval post-dates all significant commits are not bothered.
3. A detailed log is available in the **Actions** tab of the PR.

The gate resets automatically: once a reviewer approves the PR again
(triggering the `pull_request_review` event), the workflow re-evaluates and
the check turns green if no further changes are found.

## Force-pushes

If the branch is force-pushed, the commit SHA recorded in the approval may no
longer exist in the PR history.  In that case the gate **fails conservatively**
and requires re-approval, because it cannot determine the size of the change.

## Running the script locally

```bash
export GITHUB_TOKEN=ghp_...
export GITHUB_REPOSITORY=openvinotoolkit/model_server
export PR_NUMBER=1234
export MAIN_BRANCH=main
python .github/scripts/reapproval_gate.py
```

## Running the unit tests

```bash
python .github/scripts/test_reapproval_gate.py
```

The unit tests cover `is_merge_from_main()` and `find_latest_approval_per_user()` — the
two pure functions that contain the core decision logic — without requiring any
GitHub credentials.
