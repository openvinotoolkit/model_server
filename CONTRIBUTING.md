# Contributing to OpenVINO™ Model Server

Thank you for your interest in contributing to OpenVINO™ Model Server!
Below you will find the key guidelines for submitting pull requests.

## Pull Request Process

1. **Fork** the repository and create your branch from `main`.
2. Ensure your code follows the C++ style enforced by `cpplint` and
   `clang-format`.  Run `make style` to check locally.
3. All source files must carry the Apache 2.0 license header.
4. Add or update unit tests in `src/test/` for any new or changed
   functionality.
5. Update documentation in `docs/` when adding new features or changing
   existing behavior.
6. Open a pull request targeting the `main` branch.  Fill in the PR template
   and ensure all required status checks pass before requesting review.

## Reapproval Gate

This repository uses an automated **reapproval gate** to protect approved
pull requests from accumulating large unreviewed changes.

### What it does

If a PR has been approved and subsequent commits introduce **more than 10 lines
of code** (additions + deletions combined), the `reapproval-gate` required
status check fails and re-approval is required before merge.

Commits that only merge the `main` branch into your feature branch (e.g. via
GitHub's "Update branch" button) are **excluded** from this count.

### How to pass the gate

- Keep post-approval changes small (≤ 10 LOC).
- Request a fresh review after substantial updates; the gate resets
  automatically once a new approval is submitted.
- Merging `main` into your branch to resolve conflicts does **not** require
  re-approval.

### Full documentation

See [docs/reapproval-gate.md](docs/reapproval-gate.md) for a complete
description of the gate logic, configuration options, and branch protection
setup instructions.

## Code Style

- C++ style: `cpplint` + `clang-format` (`make style`)
- All source files must carry the Apache 2.0 license header
- No `using namespace std;` or `using namespace ov;`

## Building and Testing

Refer to the [build instructions](docs/build_from_source.md) and the
repository's Makefile for Docker-based build and test commands.

For C++ unit tests:
```bash
bazel test --test_summary=detailed --test_output=streamed //src:ovms_test
```

For targeted tests:
```bash
bazel test --test_filter="SuiteName.TestName" //src:ovms_test
```

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
