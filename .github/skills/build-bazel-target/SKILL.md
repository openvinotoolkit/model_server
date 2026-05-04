---
name: build-bazel-target
description: "Use when building a specific Bazel target for OVMS inside the -build Docker container. Default targets are //src:ovms (server binary) and //src:ovms_test (C++ gtest binary); also handles //src:ovms_shared (C API libovms_shared.so) and arbitrary user-supplied targets. Covers locating/starting the build container, invoking bazel build, forwarding http_proxy/https_proxy/no_proxy, switching distro to redhat via --//:distro=redhat, and reading bazel build error output. Trigger phrases: 'build ovms', 'build ovms_test', 'rebuild server', 'bazel build', '//src:ovms', '//src:ovms_test', 'compile target', 'build in container'."
---

# Build a Bazel Target in the Build Container

<!--
  USER DEFAULTS — OPTIONAL PLACEHOLDERS
  -------------------------------------
  Edit the values below to hardcode defaults for this skill. The agent MUST
  read this section at every run; if a value is set (i.e. not left as the
  literal `<...>` placeholder), use it instead of asking the user. Leave a
  value unchanged to keep the normal interactive flow for that field.

  DEFAULT_CONTAINER: <unset>      # e.g. ovms-build-fix_pull_all_dl_failed
  DEFAULT_TARGETS:   <unset>      # space-separated, e.g. //src:ovms //src:ovms_test
  DEFAULT_DISTRO:    <unset>      # ubuntu (omit flag) | redhat (adds --//:distro=redhat)
-->

Use this skill any time the user asks to **build** an OVMS Bazel target (server, tests, C API library, or any other label). For *running* tests, use the `run-single-gtest` skill instead.

## Apply user defaults (every run)

Before prompting the user for anything, parse the **USER DEFAULTS** comment block at the top of this file:

1. If `DEFAULT_CONTAINER` is set, use it verbatim and skip the "choose container" questions.
2. If `DEFAULT_TARGETS` is set and the user did not specify targets in the request, use that list as the Bazel target arguments.
3. If `DEFAULT_DISTRO` is `redhat`, add `--//:distro=redhat` to every Bazel command.
4. Otherwise (placeholder still `<unset>`), follow the normal interactive flow described in the sections below.

Always tell the user which defaults were applied (e.g. "Using `DEFAULT_CONTAINER=ovms-build-fix_pull_all_dl_failed` and `DEFAULT_TARGETS=//src:ovms //src:ovms_test` from the skill file.").

## When to use

- User asks to build, rebuild, or compile a specific Bazel target.
- User mentions `bazel build`, `//src:ovms`, `//src:ovms_test`, `//src:ovms_shared`, or another `//...` label.
- User asks to verify that recent edits compile cleanly without running tests.

## Default targets

If the user does not name a target, build **both** of these (in this order — server first, tests second; many test deps overlap with the server, so this maximizes cache reuse):

1. `//src:ovms` — main OVMS server binary
2. `//src:ovms_test` — C++ gtest binary

Other commonly requested targets:

| Target | Purpose |
|--------|---------|
| `//src:ovms_shared` | C API shared library (`libovms_shared.so`) |
| `//src:ovms` | Main server binary |
| `//src:ovms_test` | Unit test binary |

## Do NOT use

- For *running* tests — use the `run-single-gtest` skill.
- For Python functional tests (`make test_functional`).
- For full Docker image builds (`make docker_build`).

## Pre-flight

1. Confirm the workspace is the OVMS repo root (contains `WORKSPACE`, `src/`, `Makefile`).
2. Confirm the target label. If ambiguous, ask once or default to `//src:ovms` + `//src:ovms_test`.
3. For Red Hat builds, append `--//:distro=redhat` to **every** Bazel command.
4. **Resolve the image tag (`OVMS_CPP_IMAGE_TAG`) before creating a new container.** Do **not** silently fall back to `latest`.
   ```bash
   echo "${OVMS_CPP_IMAGE_TAG:-<unset>}"
   ```
   - If `OVMS_CPP_IMAGE_TAG` is **already exported** in the user's shell, reuse that value verbatim and confirm with the user.
   - Otherwise ask: "`OVMS_CPP_IMAGE_TAG` is not set. What tag should I use? (e.g. `latest-<branch>`, `latest-<feature>`)"
   - Use the resolved tag when constructing the `docker run` image argument: `openvino/model_server-build:${OVMS_CPP_IMAGE_TAG}`.

## Choose the `-build` container (ask unless `DEFAULT_CONTAINER` is set)

Builds must run **inside** the `-build` Docker container with the repo mounted at `/ovms`. **Do not auto-discover or pick a running container.** If `DEFAULT_CONTAINER` is set, use it as described above. Otherwise, ask the user one of two questions and wait for an answer before invoking `docker exec` / `docker run`:

> **Question A — reuse:** "Which existing `-build` container should I use? Please give me its name or ID (e.g. from `docker ps --format '{{.Names}}\t{{.Image}}'`)."
>
> **Question B — create:** "No container provided — should I create a new one named `ovms-build-<purpose>` from `<image>` with `-v $(pwd):/ovms`? (suggest a name that makes the purpose obvious in `docker ps`)."

Only proceed once `DEFAULT_CONTAINER` is applied, the user supplies a container name/ID, or the user confirms a new container with a specific name.

### Step A — reuse a user-specified container

Use the **name or ID the user gave you** verbatim. Do not substitute another container even if you see one in `docker ps`.

```bash
docker exec -e http_proxy="$http_proxy" \
            -e https_proxy="$https_proxy" \
            -e no_proxy="$no_proxy" \
            <user_supplied_container> \
  bash -lc 'cd /ovms && bazel build //src:ovms //src:ovms_test'
```

Single target:
```bash
docker exec -e http_proxy="$http_proxy" \
            -e https_proxy="$https_proxy" \
            -e no_proxy="$no_proxy" \
            <user_supplied_container> \
  bash -lc 'cd /ovms && bazel build //src:ovms_test'
```

### Step B — create a new, clearly named container (only after user confirms)

Always:
- Mount the **current** working directory: `-v "$(pwd)":/ovms` (run from the OVMS repo root).
- Give the container a **meaningful name** with `--name` so the user can identify it in `docker ps` (e.g. `ovms-build-<branch>`, `ovms-build-<feature>`, `ovms-redhat-build`). Confirm the name with the user before running.
- Prefer `-d` (detached) + `docker exec` for repeatable build runs, instead of `--rm` (which discards the container after one command and forces a new one for every re-run).

```bash
# Confirm with user: name="ovms-build-<purpose>", image="openvino/model_server-build:${OVMS_CPP_IMAGE_TAG}"
docker run -d -it \
  --name <agreed_container_name> \
  -v "$(pwd)":/ovms \
  -w /ovms \
  -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" -e no_proxy="$no_proxy" \
  openvino/model_server-build:${OVMS_CPP_IMAGE_TAG} bash

# Then build against that named container:
docker exec -e http_proxy="$http_proxy" \
            -e https_proxy="$https_proxy" \
            -e no_proxy="$no_proxy" \
            <agreed_container_name> \
  bash -lc 'cd /ovms && bazel build //src:ovms //src:ovms_test'
```

User can verify which container is doing what at any time with:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}'
```

## Red Hat (UBI9) builds

Add `--//:distro=redhat` to **every** Bazel command:

```bash
bazel build --//:distro=redhat //src:ovms //src:ovms_test
```

The `--//:distro=redhat` flag is **always required** in every Bazel command, regardless of whether you are running from the host or inside the container. While the `-build` container auto-detects the OS from `/etc/redhat-release` for environment setup, Bazel still requires the explicit flag to configure its build behavior.

## After the build

- **Success**: report the target(s) built and the elapsed wall time from Bazel's final `INFO: Elapsed time:` line. Built artifacts live under `bazel-bin/src/` (e.g. `bazel-bin/src/ovms`, `bazel-bin/src/ovms_test`).
- **Failure**: surface the first compiler error verbatim. If output is truncated, re-run with `--verbose_failures` against the **same user-supplied container** and tail the failing action log:
  ```bash
  docker exec -w /ovms <user_supplied_container> \
    bash -lc 'bazel build --verbose_failures //src:ovms_test 2>&1 | tail -n 200'
  ```

## Tips & pitfalls

- **Do not run `bazel` on the host.** Always go through the `-build` container so the toolchain and OpenVINO paths match.
- **Forward proxies** (`http_proxy`/`https_proxy`/`no_proxy`) for the *first* build of a fresh checkout — Bazel fetches external repos over the network. Cached builds do not need them.
- **Build the server before the tests** when both are requested; this maximizes cache reuse and surfaces production-code errors first.
- **Avoid `bazel clean`** unless the user explicitly asks — full rebuilds are very slow. Prefer `bazel build` and let Bazel handle incremental work.
- **One in-progress build at a time per workspace**: if a previous async build is still running, wait for it or kill its terminal before starting another — two concurrent Bazel servers will block each other on `bazel-out`.
- **Linker / shared-lib changes**: when modifying `BUILD.bazel`, `WORKSPACE`, or `third_party/**`, a clean rebuild of the affected target may be required.
