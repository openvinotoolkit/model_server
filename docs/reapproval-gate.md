# Reapproval Gate

The reapproval gate is an automated GitHub Actions workflow that prevents
approved pull requests from being silently merged after significant new code is
introduced without a fresh review.

## How it works

The gate runs whenever a PR is **opened, synchronized (new commits pushed),
reopened, marked ready for review**, or whenever a **review is submitted**.

```
                   PR has at least
                   one approval?
                       │
               No      │      Yes
          ─────────────┼─────────────
          ▼                          ▼
        PASS              Commits exist after
                          most recent approval?
                                │
                        No      │      Yes
                   ─────────────┼──────────────
                   ▼                           ▼
                 PASS           Calculate churn*
                                       │
                               churn > 10 LOC?
                                       │
                               No      │      Yes
                          ─────────────┼──────────
                          ▼                       ▼
                        PASS                   FAIL
                                          (re-request review)
```

\* "Churn" is **additions + deletions** across commits pushed after the most
recent approval, **excluding** merge-from-main commits (e.g. the commits
created by GitHub's "Update branch" button or `git merge main`).

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
| `MAIN_BRANCH` | `main` | Name of the default branch. Merge-from-`MAIN_BRANCH` commits are excluded from the churn calculation. |
| `LOC_THRESHOLD` | `10` | Maximum allowed additions + deletions after the most recent approval before re-approval is required. Change this constant in `.github/scripts/reapproval_gate.py`. |

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
2. The workflow attempts to **re-request review** from all users who previously
   approved the PR, using `POST /repos/{owner}/{repo}/pulls/{pr}/requested_reviewers`.
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

The unit tests cover `is_merge_from_main()` and `find_latest_approval()` — the
two pure functions that contain the core decision logic — without requiring any
GitHub credentials.
