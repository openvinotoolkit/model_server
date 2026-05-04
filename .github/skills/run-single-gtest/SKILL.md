---
name: run-single-gtest
description: "Use when running, re-running, filtering, or debugging a single OVMS C++ gtest fixture or test case from src/test/ inside the -build Docker container. Covers locating/starting the build container, invoking bazel test with --test_filter for one suite or one test, forwarding http_proxy/https_proxy/no_proxy, switching distro to redhat via --//:distro=redhat, and reading bazel-testlogs/src/ovms_test/test.log when output is truncated. Trigger phrases: 'run this test', 'run a single gtest', 'rerun failing test', 'bazel test --test_filter', 'ovms_test', 'TEST_F', 'gtest in container'."
---

# Run a Single GTest in the Build Container

<!--
  USER DEFAULTS — OPTIONAL PLACEHOLDERS
  -------------------------------------
  Edit the values below to hardcode defaults for this skill. The agent MUST
  read this section at every run; if a value is set (i.e. not left as the
  literal `<...>` placeholder), use it instead of asking the user. Leave a
  value unchanged to keep the normal interactive flow for that field.

  DEFAULT_CONTAINER:  <unset>     # e.g. ovms-build-fix_pull_all_dl_failed
  DEFAULT_TEST_FILTER: <unset>    # e.g. HfPull.ResumeShutdown  or  LibGit2LfsWipMarker.*
  DEFAULT_DISTRO:     <unset>     # ubuntu (omit flag) | redhat (adds --//:distro=redhat)
-->

Use this skill any time the user asks to run **one** gtest fixture or test case (not the full `//src:ovms_test` suite). The full suite is time-expensive — always prefer a `--test_filter` first and only widen the filter after the targeted run is green.

## Apply user defaults (every run)

Before prompting the user for anything, parse the **USER DEFAULTS** comment block at the top of this file:

1. If `DEFAULT_CONTAINER` is set, use it verbatim and skip the "choose container" questions.
2. If `DEFAULT_TEST_FILTER` is set and the user did not supply a different filter in the request, use it as the `--test_filter=` value.
3. If `DEFAULT_DISTRO` is `redhat`, add `--//:distro=redhat` to every Bazel command.
4. Otherwise (placeholder still `<unset>`), follow the normal interactive flow described in the sections below.

Always tell the user which defaults were applied (e.g. "Using `DEFAULT_CONTAINER=ovms-build-fix_pull_all_dl_failed` and `DEFAULT_TEST_FILTER=HfPull.ResumeShutdown` from the skill file.").

## When to use

- User asks to run, re-run, or debug a specific test in `src/test/**/*.cpp` (e.g. `HfPull.ResumeShutdown`, `LibGit2LfsWipMarker.*`, `HfDownloaderClassTest.Methods`).
- User mentions `bazel test`, `ovms_test`, `--test_filter`, `TEST_F`, `TEST(`, or a fixture/test name.
- A previous test run failed and the user wants the same filter re-executed.

## Do NOT use

- For running the **whole** `//src:ovms_test` suite (skip the filter, run only after targeted tests pass).
- For Python functional tests (`make test_functional`).
- For build-only requests (`bazel build //src:ovms`) — no test invocation needed.

## Pre-flight

1. Confirm the workspace is the OVMS repo root (contains `WORKSPACE`, `src/`, `Makefile`).
2. **Resolve the image tag (`OVMS_CPP_IMAGE_TAG`) before creating a new container.** Do **not** silently fall back to `latest`.
   ```bash
   echo "${OVMS_CPP_IMAGE_TAG:-<unset>}"
   ```
   - If `OVMS_CPP_IMAGE_TAG` is **already exported** in the user's shell, reuse that value verbatim and confirm with the user.
   - Otherwise ask: "`OVMS_CPP_IMAGE_TAG` is not set. What tag should I use? (e.g. `latest-<branch>`, `latest-<feature>`)"
   - Use the resolved tag when constructing the `docker run` image argument: `openvino/model_server-build:${OVMS_CPP_IMAGE_TAG}`.
3. Identify the target test:
   - Suite + test: `--test_filter="SuiteName.TestName"`
   - All tests in a suite: `--test_filter="SuiteName.*"`
   - Multiple, colon-separated: `--test_filter="A.x:B.y:C.*"`
4. If the test needs models, ensure `make prepare_models` has been run at least once. LLM model regeneration:
   ```bash
   rm -rf src/test/llm_testing && make prepare_models
   ```

## Choose the `-build` container

Tests must run **inside** the `-build` Docker container with the repo mounted at `/ovms`. **Do not auto-discover or pick a running container.** If `DEFAULT_CONTAINER` is set in the defaults block above, use it. Otherwise, ask the user one of these two questions and wait for an answer before invoking `docker exec` / `docker run`:

> **Question A — reuse:** "Which existing `-build` container should I use? Please give me its name or ID (e.g. from `docker ps --format '{{.Names}}\t{{.Image}}'`)."
>
> **Question B — create:** "No container provided — should I create a new one named `ovms-build-<purpose>` from `<image>` with `-v $(pwd):/ovms`? (suggest a name that makes the purpose obvious in `docker ps`)."

Only proceed once `DEFAULT_CONTAINER` is available, or the user supplies a container name/ID, or confirms a new container with a specific name.

### Step A — reuse a user-specified container

Use the **name or ID the user gave you** verbatim. Do not substitute another container even if you see one in `docker ps`.

```bash
docker exec -e http_proxy="$http_proxy" \
            -e https_proxy="$https_proxy" \
            -e no_proxy="$no_proxy" \
            <user_supplied_container> \
  bash -lc 'cd /ovms && bazel test \
    --test_summary=detailed --test_output=streamed \
    --test_filter="SuiteName.TestName" \
    --test_env=http_proxy="${http_proxy}" \
    --test_env=https_proxy="${https_proxy}" \
    --test_env=no_proxy="${no_proxy}" \
    //src:ovms_test'
```

### Step B — create a new, clearly named container (only after user confirms)

Always:
- Mount the **current** working directory: `-v "$(pwd)":/ovms` (run from the OVMS repo root).
- Give the container a **meaningful name** with `--name` so the user can identify it in `docker ps` (e.g. `ovms-build-<branch>`, `ovms-test-<feature>`, `ovms-redhat-build`). Confirm the name with the user before running.
- Prefer `-d` (detached) + `docker exec` for repeatable test runs, instead of `--rm` (which discards the container after one command and forces a new one for every re-run).

```bash
# Confirm with user: name="ovms-test-<purpose>", image="openvino/model_server-build:${OVMS_CPP_IMAGE_TAG}"
docker run -d -it \
  --name <agreed_container_name> \
  -v "$(pwd)":/ovms \
  -w /ovms \
  -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" -e no_proxy="$no_proxy" \
  openvino/model_server-build:${OVMS_CPP_IMAGE_TAG} bash

# Then run the filtered test against that named container:
docker exec -e http_proxy="$http_proxy" \
            -e https_proxy="$https_proxy" \
            -e no_proxy="$no_proxy" \
            <agreed_container_name> \
  bash -lc 'cd /ovms && bazel test \
    --test_summary=detailed --test_output=streamed \
    --test_filter="SuiteName.TestName" \
    --test_env=http_proxy="${http_proxy}" \
    --test_env=https_proxy="${https_proxy}" \
    --test_env=no_proxy="${no_proxy}" \
    //src:ovms_test'
```

User can verify which container is doing what at any time with:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}'
```

## Red Hat (UBI9) builds

Add `--//:distro=redhat` to **every** Bazel command (build and test):

```bash
bazel test --//:distro=redhat \
  --test_summary=detailed --test_output=streamed \
  --test_filter="SuiteName.TestName" \
  //src:ovms_test
```

The `--//:distro=redhat` flag is **always required** in every Bazel command, regardless of whether you are running from the host or inside the container. While the `-build` container auto-detects the OS from `/etc/redhat-release` for environment setup, Bazel still requires the explicit flag to configure its build behavior.

## After the run

- **Pass**: report the filter, elapsed time, and `[ PASSED ]` count from the streamed output.
- **Fail or truncated output**: read the full log directly using the **same container name/ID the user provided**:
  ```bash
  docker exec -w /ovms <user_supplied_container> \
    sh -c 'ls -la bazel-testlogs/src/ovms_test/test.log; \
           tail -n 200 bazel-testlogs/src/ovms_test/test.log'
  ```
  Verify the log mtime matches the run that just finished — earlier runs leave stale logs at the same path.

## Tips & pitfalls

- **Do not run `bazel` on the host.** Always go through the `-build` container so the toolchain and OpenVINO paths match.
- **Network-dependent tests** (HuggingFace clone, etc.) need `http_proxy`/`https_proxy`/`no_proxy` forwarded **both** to `docker exec` (`-e ...`) **and** to Bazel test workers (`--test_env=...`). Forgetting the latter causes `Could not resolve host` failures.
- **Running fixtures in parallel forks** (`EXPECT_EXIT`, child processes spawned by tests like `HfPull.ResumeTerminate`) require `--test_output=streamed` to see the live output; `--test_output=errors` will hide partial progress.
- **Avoid `--cache_test_results=no`** unless the user specifically wants to bypass Bazel's test cache; Bazel will re-run a cached green test only when its inputs change.
- **One in-progress filter at a time**: if the user requests another run, kill the previous async terminal first to avoid two concurrent Bazel servers contending for the same `bazel-out`.
