---
description: "Use when: editing, implementing, fixing, or refactoring C++ source files in the OVMS repository"
applyTo: "src/**/*.{cc,cpp,h,hpp}"
---
After completing code changes to OVMS C++ files, you MUST run the **build-validate** subagent to verify the build, run relevant tests, and check lint.

Delegation rules:
- Invoke `build-validate` as a **subagent** (this keeps build logs out of the main conversation context)
- Pass it a short description of what changed and which test fixtures are relevant (e.g. "Changed http_graph_executor_impl.cpp — run test filter HttpGraphExecutor*")
- Do this automatically — do NOT wait for the user to ask
- Do it exactly ONCE per set of changes
- Present the subagent's compact report to the user and let them decide next steps
- If the subagent reports failures, fix them and let the user decide whether to re-validate
