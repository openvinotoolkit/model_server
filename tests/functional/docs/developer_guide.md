# OpenVINO&trade; Model Server Functional Tests - Developer Guide

## Introduction

This guide explains how to run and how to write functional tests located in `tests/functional`.

Functional tests treat OpenVINO Model Server (OVMS) as a black box. A test starts a real server
(Docker container, release binary or C API library), sends requests through a public API
(KServe gRPC/REST, OpenAI, Cohere, Triton client) and verifies responses and server logs.

- [Test layout](#test-layout)
- [Prerequisites](#prerequisites)
- [Set up the environment](#set-up-the-environment)
- [Run the tests](#run-the-tests)
  - [Using make](#using-make)
  - [Using pytest directly](#using-pytest-directly)
  - [Pytest options cheat sheet](#pytest-options-cheat-sheet)
  - [Windows](#windows)
  - [Selecting tests](#selecting-tests)
  - [Run without downloads (airplane mode)](#run-without-downloads-airplane-mode)
  - [Run tests against a remote OVMS](#run-tests-against-a-remote-ovms)
  - [Check kernel messages (dmesg) on Linux](#check-kernel-messages-dmesg-on-linux)
  - [Configuration variables](#configuration-variables)
  - [Results and logs](#results-and-logs)
- [How the framework works](#how-the-framework-works)
- [Write a test](#write-a-test)
  - [Step by step](#step-by-step)
  - [Test template](#test-template)
  - [Markers](#markers)
  - [Parametrization](#parametrization)
  - [Fixtures and Context](#fixtures-and-context)
  - [Adding a model](#adding-a-model)
  - [Testing extra generative models](#testing-extra-generative-models-tt_extra_generative_models)
  - [Logging, steps and timeouts](#logging-steps-and-timeouts)
  - [Skipping tests](#skipping-tests)
  - [Marking known bugs](#marking-known-bugs)
- [Code style](#code-style)
- [Debugging](#debugging)
- [Troubleshooting](#troubleshooting)

## Test layout

```
tests/
├── requirements.txt            # Python dependencies of the functional tests
├── reservation_manager.yml     # Port pool configuration used when TT_*_STARTING_PORT are not set
└── functional/
    ├── conftest.py             # pytest hooks, registers fixtures as plugins
    ├── config.py               # all configuration read from TT_* environment variables
    ├── pylintrc                # pylint configuration for this directory
    ├── test_*.py               # test modules
    ├── constants/              # enums and constants: OVMS types, devices, requirements, components, paths
    ├── fixtures/               # pytest fixtures: context, start of OVMS, API types, common parameters
    ├── models/                 # model definitions (ModelInfo) and model collections (ModelsLib)
    ├── object_model/           # wrappers of OVMS: docker/binary/C API instances, params, config, log monitor
    ├── utils/                  # hooks, marks, logger, ports, HTTP/gRPC clients, assertions, helpers
    └── data/                   # test data, for example C API wrapper and python custom nodes
```

## Prerequisites

- Python 3.10, 3.12 or 3.14 (see the header of [tests/requirements.txt](../../requirements.txt)).
- Docker, when testing a container image (`TT_OVMS_TYPE=DOCKER`, the default). The user must be allowed to use Docker.
- An OVMS image or an OVMS release package to test:
  - Docker types: an image available locally or pullable, for example `openvino/model_server:latest`.
    Build it with `make docker_build` (see the [developer guide](../../../docs/developer_guide.md)).
  - Binary / C API types: a release package (`dist/<os>/ovms.tar.gz` or `ovms.zip`, see `TT_OVMS_C_RELEASE_ARTIFACTS_PATH`).
    Linux: build it with `make targz_package` ([Building Binary Package](../../../docs/build_from_source.md#building-binary-package)).
    Windows: create it with `windows_create_package.bat` ([Creating deployment package](../../../docs/windows_developer_guide.md#creating-deployment-package)).
    Released packages and how to use them are described in the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md).
- A Hugging Face token. Most tests download OpenVINO models from the Hub and some of them are gated.
  Provide it with `HF_TOKEN` or `HF_TOKEN_FILE_PATH` (default file: `~/ovms_tokens/huggingface_token`).
  `HF_TOKEN` takes priority. Never commit a token to the repository.
- GPU / NPU tests (see [accelerators](../../../docs/accelerators.md) for drivers and configuration):
  - Linux: the user must belong to the `render` group, the devices `/dev/dri` and `/dev/accel` must be present
    and the GPU / NPU drivers must be installed on the host.
  - Windows: the Intel GPU / NPU drivers must be installed. The tests run the binary from the release package,
    so no group membership or device files are needed.
- Enough disk space for models. Downloaded models are stored under `TT_MODELS_PATH` (default `~/ovms_models`),
  generative models under `TT_GENERATIVE_MODELS_LOCAL_PATH` (default `~/ovms_models/generative_models`).

## Set up the environment

```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

The `make test_functional` target creates the `.venv` virtual environment and installs `tests/requirements.txt` on its own.
To prepare the environment manually from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
pip3 install -r tests/requirements.txt
```

Local, personal settings (paths, tokens, image names) can be placed in a `user_config.py` file in the repository root.
`config.py` imports it before reading any variable, so it may export `os.environ["TT_..."]` values:

```python
import os

os.environ["TT_OVMS_IMAGE_NAME"] = "openvino/model_server:latest"
os.environ["TT_TARGET_DEVICE"] = "CPU"
os.environ["TT_LOGGING_LEVEL_OVMS"] = "DEBUG"
```

Do not commit secrets from this file.

## Run the tests

> **Important**: always run `pytest` from the repository root. Tests import modules as `tests.functional...`.

### Using make

```bash
make test_functional OVMS_CPP_DOCKER_IMAGE=openvino/model_server OVMS_CPP_IMAGE_TAG=latest
```

The target runs the following command inside `.venv`:

```bash
pytest --confcutdir=tests/functional --junitxml=report.xml --html=report.html --self-contained-html -v -rA -s tests/functional/
```

Use `TEST_PATH` to narrow the scope. When `TEST_PATH` is empty or `.`, `make` falls back to `tests/functional/`.

```bash
make test_functional TEST_PATH=tests/functional/test_embeddings.py
```

### Using pytest directly

```text
. .venv/bin/activate
export OVMS_CPP_DOCKER_IMAGE=openvino/model_server
export OVMS_CPP_IMAGE_TAG=latest

# everything that is enabled by default (on-commit tests)
pytest tests/functional -v

# list tests without running them
pytest tests/functional --collect-only -q

# in parallel, 4 workers
pytest tests/functional -n 4 --dist loadgroup -v

# a single module
pytest tests/functional/test_embeddings.py -v

# a single test, all parametrizations
pytest tests/functional/test_text_generation.py::TestTextGeneration::test_text_generation_endpoints -v

# a single test, only parametrizations whose id contains all the given words
pytest tests/functional/test_text_generation.py::TestTextGeneration::test_text_generation_endpoints \
    -k "CPU and LM_CB and TRACE and False" -v

# a single parametrization: the full test id in quotes (copy it from --collect-only -q)
pytest "tests/functional/test_text_generation.py::TestTextGeneration::test_text_generation_endpoints[OPENAI:REST-DOCKER-UBUNTU24-CPU-DummyCyclicGpt2OvHf-LM_CB-TRACE-temperature=0-max_tokens=10-stream=False-chat/completions]" -v
```

Test ids contain the target device, the model class name and the other parameters, so `-k` can select them,
for example `-k "CPU and LM_CB"`. The `-k` parser does not accept `=` (not even escaped), so `stream=False` cannot be used:
match the value only (`False`) or use the full test id.
To get the exact id of one parametrization, run `pytest tests/functional --collect-only -q` and copy the line.
Put the id in quotes: it contains `[`, `]` and `/`, which shells would otherwise interpret.
The exact ids depend on `TT_TARGET_DEVICE`, `TT_OVMS_TYPE`, `TT_BASE_OS` and the models in `ModelsLib`,
so the ids above may differ on your setup.

Notes:

- Quote `-k` expressions that contain spaces. In Jenkins `PYTEST_PARAMS` they must be quoted too, for example
  `tests/functional -k "multi and TRACE and VLM_CB and False"`. Unquoted words become separate arguments and pytest treats them as paths.
- `-k` matches case-insensitive substrings of the test name, parameter ids and markers.
  Use `--collect-only -q` to see the exact ids before choosing an expression.
- Which tests are collected is first decided by the `TT_*` variables (run type, device, OVMS type, ...),
  see [Selecting tests](#selecting-tests). `-k` only narrows that set.
- `-n` requires `pytest-xdist` (installed from `tests/requirements.txt`). Keep `TT_XDIST_WORKERS` equal to `-n`,
  because the framework uses it for timeouts and to validate the number of workers against the number of target devices.
  `make test_functional` does this itself: `-n $TT_XDIST_WORKERS` (`4` when it is not set).


### Windows

Windows runs always use the binary from an OVMS release package. Use the script from the repository root:

```bat
set HF_TOKEN=<your_huggingface_token>
windows_test_functional.bat c:\git\model_server\dist\windows\ovms.zip
```

Optional arguments are the pytest target with options and the number of xdist workers:

```bat
windows_test_functional.bat c:\git\model_server\dist\windows\ovms.zip "tests/functional/test_embeddings.py -k on_commit" 2
```

The script sets `TT_BASE_OS=windows`, `TT_OVMS_TYPE=BINARY`, `TT_DISABLE_DMESG_LOG_MONITOR=True` and `TT_XDIST_WORKERS=4`, creates `.venv` and installs `tests\requirements.txt`. It runs pytest with
`-n <workers> --dist loadgroup` and writes `report.xml` (JUnit) and `report.html` (self-contained HTML report)
in the repository root. See [Results and logs](#results-and-logs).
Other `TT_*` variables are taken from the current environment. See also the
[Windows developer guide](../../../docs/windows_developer_guide.md).

### Selecting tests

What is executed is the result of several filters. A test is deselected during collection when any of them rejects it.

| Filter | How |
|--------|-----|
| Run type | `TT_ON_COMMIT_TESTS` (default `True`) enables tests marked `api_on_commit`; `TT_RUN_REGRESSION_TESTS` (default `False`) enables `api_regression` |
| Device | `TT_TARGET_DEVICE`, for example `CPU` or `CPU,GPU,NPU` |
| OVMS type | `TT_OVMS_TYPE`, for example `DOCKER`, `BINARY`, `BINARY_DOCKER`, `CAPI`, `CAPI_DOCKER`, `DOCKER_CMD_LINE` |
| Base OS | `TT_BASE_OS`: `ubuntu22`, `ubuntu24`, `redhat`, `windows`; a comma-separated list iterates tests over all of them |
| Priority | `TT_TESTS_PRIORITY_LIST`, for example `high,medium` |
| Requirement | `TT_REQUIREMENTS` / `TT_EXCLUDE_REQUIREMENTS` (values from `constants/requirements.py`) |
| Component | `TT_COMPONENTS` / `TT_EXCLUDE_COMPONENTS` |
| Name | pytest `-k` or `TT_PYTEST_KEYWORD_FILTER` (case insensitive substring of the test name) |

Requirements cannot be included and excluded at once, the same applies to components.

### Run without downloads (airplane mode)

By default, the start of a test session downloads Hugging Face models, pulls Docker images and checks files that
already exist. Set `TT_AIRPLANE_MODE=True` when everything needed is already on the machine, or when the machine has no
access to the Internet or to the registry. In this mode the framework:

- skips downloading and updating models (`download_resources_master` in `utils/hooks.py`),
- skips pulling Docker images, and uses the image already present locally instead,
- does not connect to remote resources.

Because nothing is fetched, everything the selected tests need must already be present:

- the models in `TT_MODELS_PATH` and `TT_GENERATIVE_MODELS_LOCAL_PATH`,
- the OVMS image (`docker images`) or the release package (`TT_OVMS_C_RELEASE_ARTIFACTS_PATH`),
- Python packages from `tests/requirements.txt`.

A missing model or image is not downloaded; the test fails instead.

Typical flow: run the tests once with the default settings so that models and image are downloaded, then repeat with
`TT_AIRPLANE_MODE=True`:

```text
export TT_AIRPLANE_MODE=True
pytest tests/functional/test_embeddings.py -v
```

```text
set TT_AIRPLANE_MODE=True
windows_test_functional.bat c:\git\model_server\dist\windows\ovms.zip
```

`TT_OVMS_IMAGE_LOCAL=True` is narrower: it only says that the image exists locally and must not be pulled, while models
are still downloaded.

### Run tests against a remote OVMS

The framework can use an OVMS that is already running, on this or on another machine, instead of starting its own
(container, binary or C API). Set at least one of these variables, which switches the framework to the external server mode:

| Variable | Default | Description |
|----------|---------|-------------|
| `TT_OVMS_REST_PORT` | not set | REST port of the running server |
| `TT_OVMS_GRPC_PORT` | not set | gRPC port of the running server |
| `TT_SERVER_ADDRESS` | `localhost` | Address of the machine where the server runs |

When `TT_OVMS_REST_PORT` or `TT_OVMS_GRPC_PORT` is set, `start_ovms(...)` does not start anything and returns an
`OvmsExternal` object (`object_model/ovms_external.py`) that only knows the ports. Tests then send requests to
`TT_SERVER_ADDRESS:<port>`. `REMOTE_SERVER_ADDRESS`, when set in the environment, takes priority over `TT_SERVER_ADDRESS`.
A non-local `TT_SERVER_ADDRESS` is appended to `no_proxy` automatically (`TT_SET_NO_PROXY=False` disables the `NO_PROXY` update).

Example - server on `127.0.0.1` serving the models that the selected tests use, tests run from another machine:

```text
export TT_SERVER_ADDRESS=127.0.0.1
export TT_OVMS_REST_PORT=8000
export TT_OVMS_GRPC_PORT=9000
export TT_AIRPLANE_MODE=True          # nothing to download or pull on the client machine
pytest tests/functional/test_text_generation.py -k "CPU and chat/completions" -v
```

Requirements and limitations:

- **Prepare the server yourself.** The framework does not load, unload or change models on a remote server. The server
  must already serve the models used by the selected tests under the names the tests request (the model name from
  `models/`, for example `OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov`). Use `--collect-only -q` to see which models
  the selected tests use.
- **Only the ports you set are available.** A test that needs the gRPC port fails when only `TT_OVMS_REST_PORT` is set,
  and the other way round. Set both for suites that use both protocols.
- **No control over the server.** `OvmsExternal` only provides `get_port(...)`. Steps that need the server itself
  (log monitor, restart, config reload, unloading models) cannot run. `GenerativeAIUtils.unload_model_and_verify`
  skips its unload step for an external server; tests that call such methods directly are not suitable for this mode.
- **Device and OVMS type filters still apply to collection**, so set `TT_TARGET_DEVICE` to the device the remote server
  uses to select the matching test parametrizations.
- **One shared server.** All tests use the same instance. Prefer `TT_XDIST_WORKERS=0` (no parallelism) unless the server is
  prepared for concurrent load.

### Check kernel messages (dmesg) on Linux

With `TT_DISABLE_DMESG_LOG_MONITOR=False` the framework reads `dmesg` after the tests and fails a test when the kernel
reports a problem related to OVMS (segfault, general protection fault, bad RIP value, divide error, GPU hang, OOM kill).
The monitor is disabled by default, because reading `dmesg` is restricted on most distributions. Only Linux hosts are
supported; `windows_test_functional.bat` always disables it.

The monitor runs `dmesg` without `sudo`, so the kernel must allow unprivileged users to read the kernel log.
Without it the tests fail with `Missing permissions to use dmesg without sudo. Please set 'sudo sysctl -w kernel.dmesg_restrict=0'`
(or `dmesg: read kernel buffer failed: Operation not permitted`).

1. Check the current value (`1` means restricted):

   ```text
   sysctl kernel.dmesg_restrict
   dmesg -T --level=err    # must print messages instead of "Operation not permitted"
   ```

2. Allow reading the kernel log. This lasts until the next reboot:

   ```text
   sudo sysctl -w kernel.dmesg_restrict=0
   ```

   To keep the setting after a reboot:

   ```text
   echo "kernel.dmesg_restrict = 0" | sudo tee /etc/sysctl.d/99-dmesg.conf
   sudo sysctl --system
   ```

3. Enable the monitor:

   ```bash
   export TT_DISABLE_DMESG_LOG_MONITOR=False
   ```

When `TT_CLEANUP_ENVIRONMENT_ON_STARTUP=True` is also set, the session start clears the kernel buffer with `sudo dmesg -C`,
so the user also needs passwordless `sudo` for that command, for example a `/etc/sudoers.d/dmesg` entry
(edit it with `sudo visudo -f /etc/sudoers.d/dmesg`):

```
<user> ALL=(root) NOPASSWD: /usr/bin/dmesg
```

Relaxing `kernel.dmesg_restrict` lets every local user read the kernel log, which can contain kernel addresses and
hardware details. Do it only on dedicated test machines. Containers share the host kernel log, so this setting is
needed on the host, not inside the OVMS container.

### Configuration variables

All variables are documented next to their definitions in [config.py](../config.py). The most frequently used ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `OVMS_CPP_DOCKER_IMAGE` / `OVMS_CPP_IMAGE_TAG` | `openvino/model_server` / `latest` | Image under test (name and tag), compatible with the `make` parameters |
| `TT_OVMS_IMAGE_NAME` | not set | Full image name, overrides the two variables above |
| `TT_OVMS_IMAGE_LOCAL` | `False` | The image exists only locally, do not pull it |
| `TT_AIRPLANE_MODE` | `False` | Do not download models or pull images, use only local resources, see [airplane mode](#run-without-downloads-airplane-mode) |
| `TT_OVMS_REST_PORT` / `TT_OVMS_GRPC_PORT` | not set | REST / gRPC port of an already running OVMS. Setting any of them enables the external server mode, see [remote OVMS](#run-tests-against-a-remote-ovms) |
| `TT_SERVER_ADDRESS` | `localhost` | Address of the OVMS server used by the tests (overridden by `REMOTE_SERVER_ADDRESS`) |
| `TT_OVMS_TYPE` | `DOCKER` | How OVMS is started |
| `TT_TARGET_DEVICE` | `CPU` | Target device(s) |
| `TT_BASE_OS` | `ubuntu24` | OS of the tested image or package |
| `TT_OVMS_C_RELEASE_ARTIFACTS_PATH` | `dist/<os>/ovms.tar.gz` (`dist\windows\ovms.zip` on Windows) | Release package(s) for `BINARY` / `CAPI` types. Comma-separated list, one entry per `TT_BASE_OS`; local and remote locations are supported |
| `TT_ON_COMMIT_TESTS` / `TT_RUN_REGRESSION_TESTS` | `True` / `False` | Run types to execute |
| `TT_XDIST_WORKERS` | `4` | Number of xdist workers. Must match `-n`, see [Using pytest directly](#using-pytest-directly) |
| `TT_LOGGING_LEVEL_OVMS` | `INFO` | OVMS log level, passed to the server as `--log_level`. Values: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR` (upper case). Tests that set their own level override it |
| `TT_LOGGING_LEVEL` | `INFO` | Log level of the test framework (Python `logging`). Values: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG` (upper case). `DEBUG` also raises the number of log lines checked when waiting for OVMS messages |
| `TT_MODELS_PATH` | `~/ovms_models` | Local models repository |
| `TT_GENERATIVE_MODELS_LOCAL_PATH` | `<TT_MODELS_PATH>/generative_models` | Local copies of generative models |
| `TT_EXTRA_GENERATIVE_MODELS` | not set | Comma-separated list of extra LLM / VLM definitions added to the generative test collections, see [extra generative models](#testing-extra-generative-models-tt_extra_generative_models) |
| `HF_TOKEN` / `HF_TOKEN_FILE_PATH` | not set / `~/ovms_tokens/huggingface_token` | Hugging Face token |
| `TT_PORTS_POOL_SIZE`, `TT_GRPC_OVMS_STARTING_PORT`, `TT_REST_OVMS_STARTING_PORT` | generated | Ports used by started servers. When not all three are set, a pool is created using `tests/reservation_manager.yml` |
| `TT_WAIT_FOR_MESSAGES_TIMEOUT` | `180` | Timeout in seconds of waiting for log messages from OVMS |
| `TT_DISABLE_DMESG_LOG_MONITOR` | `True` | Do not check `dmesg` for errors. Checking requires access to `dmesg` without `sudo`, see [dmesg](#check-kernel-messages-dmesg-on-linux) |
| `TT_TEARDOWN_DOCKER_CONTAINERS` | `False` | Remove stopped containers at the end of the session |
| `TEST_TIMEOUT` | `15` | Whole session timeout in hours |
| `MEDIAPIPE_DISABLE` | `0` | Set to `1` when the tested server was built without MediaPipe. Tests that require it are skipped |
| `TT_LANGUAGE_MODELS_ENABLED` | `True` | Set to `False` to skip generative tests |

### Results and logs

| Artifact | Location |
|----------|----------|
| JUnit report | `report.xml` (`make`, `windows_test_functional.bat` and the Jenkins pipelines) |
| HTML report | `report.html`, self-contained, can be opened or archived on its own |
| Test logs | `TEST_LOG_DIR`, default `./test_log`, one file per test |
| Artifacts | `BUILD_LOGS`, default `tests/functional/test_log_build` |
| Live log in console | use `-s` or `-o log_cli=true` |

## How the framework works

1. **`conftest.py`** registers the fixture modules from `fixtures/` as pytest plugins and implements the pytest hooks.
   The hooks live in `utils/hooks.py`.
2. **Session start (master process)**: validates the port pool, prepares artifacts and the OVMS release package (for binary and
   C API types), downloads models from Hugging Face and pulls Docker images (unless `TT_AIRPLANE_MODE=True`).
3. **Collection**: `parametrize_tests` adds parameters `target_device`, `ovms_type`, `base_os` and `model_type`
   (according to the fixtures used by the test and the markers). Then tests are filtered (see [Selecting tests](#selecting-tests)),
   the default timeout is set from the run type and the order of tests is shuffled with a fixed seed.
4. **Test execution**: the autouse `context` fixture creates a `Context` object (device, OVMS type, image or binary, port managers,
   cleanup list). The test starts the server with `start_ovms(...)`, communicates with it and the `Context` cleans up everything
   at the end, even when the test fails.
5. **Session end**: ports reservation, temporary directories and, optionally, Docker images and processes are cleaned up.

Main building blocks:

| Component | Purpose |
|-----------|---------|
| `fixtures/ovms.py` | `context` fixture, session timeout check, test environment |
| `fixtures/server.py` | `start_ovms(...)` - starts OVMS as Docker container, binary or C API according to `context.ovms_type` |
| `fixtures/api_type.py` | `*_api_type` fixtures: API client = serving (KFS, OpenAI, Cohere, Triton) + communication (gRPC, REST) |
| `object_model/ovms_params.py` | `OvmsParams` - server parameters (models, config, ports, log level, ...) |
| `object_model/ovms_instance.py` | `OvmsRunContext` and the common interface of a running OVMS (`ensure_started`, `get_port`, `unload_all_models`, `create_log`) |
| `object_model/ovms_docker.py` | `OvmsDockerInstance` and `OvmsCmdLineDockerInstance` - OVMS in a Docker container (`DOCKER`, `DOCKER_CMD_LINE`) |
| `object_model/ovms_binary.py` | `start_binary_ovms(...)` and `OvmsBinary` - OVMS started as a process from the unpacked release package (`BINARY`) |
| `object_model/ovms_log_monitor.py` | Waiting for and checking server log messages |
| `object_model/inference_helpers.py` | Helpers that run and validate inference, for example `run_llm_inference` |
| `models/` | Model definitions and collections (`ModelsLib`) |
| `constants/` | OVMS types, devices, OS, requirements ids, components, paths, expected messages |
| `utils/generative_ai/utils.py` | `GenerativeAIUtils` - prepares model, starts server and unloads model for generative tests |
| `utils/assertions.py` | Assertion helpers, for example `assert_raises_exception` |

## Write a test

### Step by step

1. Choose the module. Add a test to an existing `test_<feature>.py`, or create a new module and a class `Test<Feature>`.
2. Add the license header (Apache 2.0, current year) at the top of every new `.py` file.
3. Mark the class or the test with the markers described in [Markers](#markers). Every test needs a run type marker
   (`api_on_commit`, `api_regression`, ...) - collection fails with `Test do not have test_type` otherwise.
4. Parametrize the test, see [Parametrization](#parametrization). Request the `context` fixture and the API type fixture.
5. Describe the test in a docstring in the format used in this repository and split the body into `step(...)` calls.
6. Start OVMS with `start_ovms(...)` (or a helper like `GenerativeAIUtils.prepare_resources`) and use the returned `OvmsRunContext`.
   Do not clean up manually; every object which has a `cleanup()` method and was registered in the `context` is cleaned up automatically.
7. Run the test locally with `-k` and run `pylint` (see [Code style](#code-style)).
8. If the test needs a new model, requirement or component, add it first (see [Adding a model](#adding-a-model)).

### Test template

```python
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (full header as in other files)
#

import pytest

from tests.functional.constants.components import OvmsComponents
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.requirements import Requirements
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.target_device_configuration import nginx_mtls_not_supported_for_test
from tests.functional.models.models_library import ModelsLib
from tests.functional.object_model.inference_helpers import run_llm_inference
from tests.functional.utils.context import Context
from tests.functional.utils.generative_ai.utils import calculate_generative_test_timeout, GenerativeAIUtils
from tests.functional.utils.inference.serving.openai import OpenAIWrapper
from tests.functional.utils.logger import get_logger, step
from tests.functional.utils.test_framework import (
    skip_if_language_models_not_enabled,
    skip_if_mediapipe_disabled,
)

logger = get_logger(__name__)


@pytest.mark.priority_high
@pytest.mark.components(OvmsComponents.OVMS)
@pytest.mark.reqids(Requirements.llm)
@pytest.mark.ovms_types_supported_for_test(OvmsType.DOCKER, OvmsType.BINARY)
@skip_if_language_models_not_enabled()
@nginx_mtls_not_supported_for_test()
@skip_if_mediapipe_disabled()
class TestMyFeature:

    @pytest.mark.api_on_commit
    @pytest.mark.devices_supported_for_test(TargetDevice.CPU)
    @pytest.mark.model_type(ModelsLib.various_mini_large_language_models)
    @pytest.mark.parametrize("endpoint", [OpenAIWrapper.CHAT_COMPLETIONS])
    @pytest.mark.timeout(calculate_generative_test_timeout(480))
    def test_my_feature(self, context: Context, model_type, openai_rest_api_type, endpoint):
        """
        <b>Description:</b>
        Short description of what is verified.

        <b>Input data:</b>
        - Language model

        <b>Expected results:</b>
        OVMS loads the model and returns a valid response.

        <b>Steps:</b>
        1. Prepare language model instance
        2. Start OVMS
        3. Run simple inference
        4. Unload model
        5. Verify model is unreachable
        """
        model, result, port, request_params = GenerativeAIUtils.prepare_resources(
            context, model_type, openai_rest_api_type, endpoint,
        )

        step("Run simple inference")
        run_llm_inference(model, openai_rest_api_type, port, endpoint, request_parameters=request_params)

        GenerativeAIUtils.unload_model_and_verify(model, result, port, openai_rest_api_type, endpoint, request_params)
```

`GenerativeAIUtils.prepare_resources` creates the model object, starts OVMS with `start_ovms(...)` and returns
`(model, run_context, port, request_params)`. For tests that need custom server parameters call `start_ovms` directly:

```python
from tests.functional.fixtures.server import start_ovms
from tests.functional.object_model.ovms_params import OvmsParams

result = start_ovms(context, OvmsParams(models=[model], use_config=True, log_level="DEBUG"))
port = result.ovms.get_port(api_type)
```

### Markers

Markers are registered in `utils/marks.py` (`MarksRegistry`). A marker can be put on a class (inherited by all tests) or on a test.

| Marker | Required | Meaning |
|--------|----------|---------|
| `api_on_commit`, `api_regression`, `api_regression_single`, `api_regression_weekly`, `api_regression_weekly_single`, `api_enabling`, `api_smoke`, `api_long`, `api_performance`, `api_stress_and_load`, `api_stress_and_load_single`, `api_unstable`, `manual`, `component` | yes, one of them | Run type. Decides in which test run the test is executed and its default timeout. Currently only `api_on_commit` and `api_regression` can be enabled by `TT_ON_COMMIT_TESTS` / `TT_RUN_REGRESSION_TESTS` |
| `priority_high`, `priority_medium`, `priority_low` | yes | Priority, filtered with `TT_TESTS_PRIORITY_LIST` |
| `components(OvmsComponents.OVMS)` | yes | Component, filtered with `TT_COMPONENTS` |
| `reqids(Requirements.xxx, ...)` | yes | Requirements covered by the test, filtered with `TT_REQUIREMENTS`. Values come from `constants/requirements.py` |
| `ovms_types_supported_for_test(...)` / `ovms_types_not_supported_for_test(...)` | recommended | OVMS types the test is designed for |
| `devices_supported_for_test(...)` / `devices_not_supported_for_test(...)` | recommended | Target devices the test is designed for. Actual devices are selected by `TT_TARGET_DEVICE` and by the model collection passed to `model_type` |
| `os_types_supported_for_test(...)` / `os_types_not_supported_for_test(...)` | optional | OS types the test is designed for |
| `model_type(collection)` | for model tests | Parametrizes the test with models, see below |
| `model_aux_type(collection)` | optional | Second model (for example a draft or reranker model) from the same device |
| `all_models`, `many_models`, `iteration_info`, `input_shape`, `plugin_config` | optional | Other parametrization helpers handled in `utils/hooks.py` |
| `conditional_run_type(...)`, `conditional_run_type_by_model(...)` | optional | Assign a run type depending on device, OS or model |
| `bugs(...)` | optional | Mark a test with a known bug, see [Marking known bugs](#marking-known-bugs) |
| `timeout(seconds)` | optional | Overrides the default timeout. Without it the timeout is set from the run type, and multiplied for GPU, NPU and AUTO/HETERO |

Components and requirement ids are inherited from the class when the test does not set its own.

### Parametrization

Parameters are generated as a cartesian product of everything the test requests.

- **Target device** - the `target_device` parameter is added automatically when the test uses the `context` fixture or
  `model_type`. Values come from `TT_TARGET_DEVICE`.
- **OVMS type and base OS** - `ovms_type` and `base_os` are added from `TT_OVMS_TYPE` and `TT_BASE_OS`.
- **Models** - `@pytest.mark.model_type(collection)` adds the `model_type` parameter. The collection is a property of `ModelsLib`:
  either a list of model classes or a dict `{TargetDevice: [model classes]}`. Test ids look like `CPU-<ModelClassName>`.
- **API type** - request a fixture from `fixtures/api_type.py`, for example `openai_rest_api_type`, `cohere_rest_api_type`,
  `kfs_grpc_api_type`, `kfs_rest_api_type`, `kfs_api_type`, `triton_grpc_api_type`, `api_type`.
- **Other** - use `@pytest.mark.parametrize(..., ids=lambda x: f"name={x}")` for any additional values so that the ids are readable
  and can be used with `-k`.

Contradictory combinations are removed in `generic_test_deselect` (`utils/hooks.py`), for example the `completions` endpoint
is not used with vision language models.

### Fixtures and Context

- `context` (autouse, function scope) - holds `target_device`, `ovms_type`, `base_os`, `ovms_image`, port managers and the list
  of objects to clean up. Always accept it as the first test argument and pass it to helpers.
- `start_ovms(context, parameters, ...)` returns `OvmsRunContext`, where:
  - `result.ovms` - running server (`get_port(api_type)`, `unload_all_models()`, `create_log(...)`, `ensure_started(...)`),
  - `result.models` - models served by the instance.
- `add_test_finalizer` - register an additional cleanup callable: `add_test_finalizer(lambda: ...)`.
- `Context.test_objects` - append any object with a `cleanup()` method, it is called in reverse order at the end of the test.
- Ports are allocated by `context.port_manager_grpc` / `context.port_manager_rest`; never hard-code ports,
  so tests can run in parallel.

Add new, shared fixtures to a module in `fixtures/` and register that module in `pytest_plugins` in `conftest.py`.

### Adding a model

1. Define a model class in `models/models_generative.py` (generative models) or `models/models.py`.
   Classes are dataclasses derived from `GenerativeModel` / `ModelInfo` with fields such as `name` (Hugging Face repository),
   `precision`, `is_llm`, `is_feature_extraction`, `is_rerank`, `is_image_generation`, `is_audio`, `is_agentic`.
2. Add the class to a collection in `models/models_library.py` (`ModelsLib`). Collections are plain properties:
   a list of classes, or a `defaultdict(list, {TargetDevice.X: [...]})` when models differ per device.
3. Use the collection in `@pytest.mark.model_type(ModelsLib.<collection>)`.

All models from all `ModelsLib` collections for the selected `TT_TARGET_DEVICE` are downloaded in the session start
(`download_models` in `utils/hooks.py`), so add only the models that are really used. Downloads are incremental: files that are
missing or have a different size are downloaded again.

### Testing extra generative models

`TT_EXTRA_GENERATIVE_MODELS` adds LLM / VLM models to the existing generative test collections without changing the code.
The models are appended to `ModelsLib.various_mini_large_language_models` and `ModelsLib.various_mini_vision_language_models`,
so every test that uses them (including the on-commit combined collection) is also parametrized with the extra models.

The value is a comma-separated list of definitions in the format:

```
class_name:model_name:precision:precision_dir
```

| Field | Description |
|-------|-------------|
| `class_name` | Name of the generated model class. It appears in the test id. Include `VL` in the name to create a vision language model (`VisionLanguageModel`); otherwise a `LargeLanguageModel` is created |
| `model_name` | Hugging Face repository of the model |
| `precision` | Model precision, for example `INT4` |
| `precision_dir` | Directory name used for the precision in the local models path (`<precision_dir>/<model_name>` under `TT_GENERATIVE_MODELS_LOCAL_PATH`) |

If `class_name` matches a class already defined in `models/models_generative.py`, that class is added to the collections and the
remaining fields can be omitted (a definition with only the class name). Any other number of fields is rejected with `ValueError`.

Examples:

```text
# one new LLM
export TT_EXTRA_GENERATIVE_MODELS="MetaLlama318BInstructInt4SymCw:meta-llama/Llama-3.1-8B-Instruct:INT4:INT4_SYM_CW"

# a new VLM and an already defined class
export TT_EXTRA_GENERATIVE_MODELS="MyModelVLInt4:org/my-vlm:INT4:INT4_SYM_CW,Gemma34bItInt4CwOvHf"

pytest tests/functional/test_text_generation.py -k "MetaLlama318BInstructInt4SymCw" -v
```

The variable can also be set in `user_config.py`. The extra models are prepared like the other generative models, so they are
downloaded at the session start (unless `TT_AIRPLANE_MODE=True`) and gated models need a Hugging Face token.
Use `pytest --collect-only -q` to check the generated test ids.

### Logging, steps and timeouts

```python
from tests.functional.utils.logger import get_logger, step

logger = get_logger(__name__)

step("Start OVMS")            # marks a test step in the log, use the names from the docstring
logger.info("some details")   # regular log message
```

- Use `get_logger(__name__)`, not `print` and not `logging.getLogger`.
- Every test writes its own log file in `TEST_LOG_DIR`. Sensitive keys can be masked when `TT_STRIP_SENSITIVE_DATA=True`.
- Loading large models can take long. For generative tests use `calculate_generative_test_timeout(<test time in seconds>)`
  that adds the model initialization time.
- Prefer log monitor helpers (`result.ovms.create_log(...)`, `OvmsMessages` in `constants/ovms_messages.py`)
  to sleeping while waiting for server events.

### Skipping tests

Use helpers from `utils/test_framework.py`:

| Helper | Use |
|--------|-----|
| `skip_if_language_models_not_enabled()` | decorator, skip when `TT_LANGUAGE_MODELS_ENABLED=False` |
| `skip_if_mediapipe_disabled()` | decorator, skip when `MEDIAPIPE_DISABLE=1` |
| `nginx_mtls_not_supported_for_test()` | decorator, skip for nginx mTLS images |
| `skip_if(condition, msg)` | decorator form with any condition |
| `skip_if_runtime(condition, msg)` | call inside a test or fixture |
| `skip_not_implemented()` | decorator for a placeholder test |

### Marking known bugs

Attach the id of the tracking issue (for example `CVS-123456`) to a test with `@pytest.mark.bugs(...)`.
The marker is registered in `utils/marks.py` (`MarkBugs`). The ids are read with
`MarkBugs.get_all_bug_marks_values_from_item(item)`, so reports and hooks can use them.

> **Note**: the marker only records the bug. It does not skip, `xfail` or change the result of the test, and no hook in
> this repository acts on it. To keep a test that is blocked by a bug from failing the run, also skip it
> (see [Skipping tests](#skipping-tests)) and put the same id in the skip message.

```python
# a bug that affects the test on every device and OS
@pytest.mark.bugs("CVS-123456")

# only when the condition is true; it is evaluated when the module is imported, so use values from config.py
@pytest.mark.bugs("CVS-123456", OsType.Windows in base_os)

# only for parametrizations whose id matches: a compiled regex (matched from the beginning of the id) or the full id
@pytest.mark.bugs("CVS-123456", True, re.compile(r"GPU-.*"))

# several bugs: stack the markers
@pytest.mark.bugs("CVS-123456")
@pytest.mark.bugs("CVS-123457")
```

Rules:

- Positional arguments are `(bug_id, condition, params)`; the last two are optional. A list of entries is also accepted,
  but only the first matching entry is used, so use stacked markers for several bugs.
- Set the marker on the test function. Markers set on a class are not read by `MarkBugs`.
- Use the Jira issue id as it is written in the tracker, the same way as in code comments (`# fix for CVS-126060`).
- Remove the marker (and the skip) when the bug is fixed.

Example of a test blocked by a bug:

```python
@pytest.mark.bugs("CVS-123456")
@skip_if(True, msg="CVS-123456 OVMS crashes when the model is unloaded")
def test_unload_model(self, context, model_type):
    ...
```

When the bug affects only some parametrizations, skip inside the test instead:
`skip_if_runtime(context.target_device == TargetDevice.GPU, msg="CVS-123456 ...")`.

## Code style

- Follow the existing style of the file. The line limit and the other rules are in [pylintrc](../pylintrc).
- Run pylint from the repository root (the same command is run in the CI):

```bash
python3 -m pylint --rcfile=tests/functional/pylintrc tests/functional
```

- Every new file needs the Apache 2.0 license header with the current year.
- Pass non-trivial values through helpers and constants (`constants/`) instead of repeating literals.
- Keep tests independent. A test must not rely on the state or the order of other tests, because tests are shuffled and can run in parallel.
- Prefer existing helpers (`GenerativeAIUtils`, `run_llm_inference`, `assert_raises_exception`) to a new implementation.

## Debugging

Run a single test and print everything to the console:

```bash
pytest tests/functional/test_embeddings.py -k "CPU and float" -s -o log_cli=true -o log_cli_level=DEBUG
```

Useful variables: `TT_LOGGING_LEVEL_OVMS=DEBUG` (server log), `TT_LOGGING_LEVEL=DEBUG` (framework log),
`TT_AIRPLANE_MODE=True` (skip downloads when everything is already local), `TT_OVMS_IMAGE_LOCAL=True`.

...

When a test fails, check in this order: the test log file in `TEST_LOG_DIR`, the OVMS container log
(`docker logs <container>`; containers are removed in cleanup unless `TT_TEARDOWN_DOCKER_CONTAINERS` is used otherwise), and `dmesg`
messages when `TT_DISABLE_DMESG_LOG_MONITOR=False`.

## Troubleshooting

| Problem | Cause and solution |
|---------|--------------------|
| `PermissionError: ... bazel-out` during collection | pytest scanned the whole repository. Pass an explicit path (`pytest tests/functional/`); `make test_functional` already falls back to `tests/functional/` for an empty or `.` `TEST_PATH` |
| `ModuleNotFoundError: tests` | `pytest` is not run from the repository root |
| `Provide huggingface_token with either HF_TOKEN or HF_TOKEN_FILE_PATH envs` | A gated model cannot be downloaded. Set `HF_TOKEN` or `HF_TOKEN_FILE_PATH` |
| Image pull fails or the image is not found | Set `TT_OVMS_IMAGE_LOCAL=True` for a locally built image or pass a valid `OVMS_CPP_DOCKER_IMAGE` / `OVMS_CPP_IMAGE_TAG` |
| `Mismatch between config base os ... host os` | Binary and C API types require the host OS to equal `TT_BASE_OS` |
| All tests are deselected | Check the run type variables, `TT_TARGET_DEVICE`, `-k` filter and requirement / component filters. Use `--collect-only` |
| Tests skipped because of MediaPipe | The server was built with `MEDIAPIPE_DISABLE=1`, or the variable is set in the environment |
| `Missing permissions to use dmesg without sudo` | `TT_DISABLE_DMESG_LOG_MONITOR=False` but `kernel.dmesg_restrict=1`. Set it to `0` or disable the monitor, see [dmesg](#check-kernel-messages-dmesg-on-linux) |
| GPU / NPU tests fail to start the container | Verify `/dev/dri`, `/dev/accel` and membership of the `render` group |
| Port conflicts in parallel runs | Do not hard-code ports; use `context.port_manager_*`. Check `TT_PORTS_POOL_SIZE` and the starting ports |
