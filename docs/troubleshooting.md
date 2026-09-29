# Troubleshooting {#ovms_docs_troubleshooting}

## Introduction
This document gives information about troubleshooting the following issues while using the OpenVINO&trade; Model Server:
* [Generative models troubleshooting](#generative-models-troubleshooting)
* [Classic Model Import Issues](#model-import-issues)
* [Client Request Issues](#client-request-issues)
* [Resource Allocation](#resource-allocation)
* [Usage Monitoring](#usage-monitoring)
* [Configuring S3 storage For Use With a Proxy](#configuring-s3-storage-for-use-with-a-proxy)
* [Using GCS storage behind a proxy](#using-gcs-model-behind-a-proxy)
* [Model Cache Issues](#model-cache-issues)
* [Pulling Models from Hugging Face Hub](#pulling-models-from-hugging-face-hub)


## Generative Models Troubleshooting

Start the server in debug mode with `--log_level DEBUG`. This exposes additional details about model loading, runtime configuration, request processing, and the exact point where a failure occurs.

- Some models do not support every target device. For example, certain quantization formats are not supported on GPU (`nf4`) and some are not supported on NPU (`int4` without channel-wise quantization).
    If a model fails to load on one device, verify whether the model card or export configuration lists device-specific constraints.

- Some newer models may not be supported on older platforms or older OpenVINO runtimes.
    Xe+ platforms usually provide good coverage, but there can still be exceptions depending on the exact model architecture, precision, or plugin version.
    If you see compatibility errors, confirm the model export and runtime version first.

- Newly enabled LLM/VLM models may not always support the continuous batching pipeline, especially in preview or early-access scenarios.
    In that case, switch to the non-continuous-batching pipeline with `--pipeline_type LM`, `--pipeline_type VLM`, or `--pipeline_type OMNI` as appropriate.
    Continuous batching variants are the defaults: `LM_CB`, `VLM_CB`, and `OMNI_CB`.
    If the model loads only after changing the pipeline type, the issue is usually a pipeline-model compatibility mismatch rather than a corrupted model.
    NPU device can't use continuous batching pipeline.
    
- An out-of-resources error during loading usually indicates insufficient RAM or VRAM for the selected model, precision, or cache configuration.
    In that case, try one of the following:
    - use a smaller model
    - choose a lower-precision quantization
    - reduce KV cache size or other memory-intensive generation parameters
    - move the workload to a device with more available memory

- LLM models deployed on NPU can consume a lot of memory when configured for long-context inputs.
    If the workload requires long prompts, large context windows, or many concurrent sessions, GPU or CPU may be a better fit.
    When NPU memory pressure appears only at long context sizes, lowering the context window or cache size may also help.

## Pulling Models from Hugging Face Hub

When pulling models from Hugging Face Hub, keep the following in mind:

- Make sure proxy configuration is correct if the environment requires a proxy.
- Confirm that `source_model` uses the `<organization>/<model_id>` format.
- Use models prepared in IR format with `.xml` and `.bin` files when applicable. PyTorch checkpoints and safetensors can be used directly when supported by the loader. Models exported and formatted via optimum-intel must be placed in the repository root.
- If a download was interrupted, rerun it. OVMS resumes from the last downloaded content.
- Check known issues in the release notes or in the relevant demos and documentation.

## Classic Model Import Issues

OpenVINO&trade; Model Server can load models in two layouts:
- Versioned layout: model files are stored in numerical subdirectories and selected according to [version policy](./model_version_policy.md).
- Non-versioned layout: model files are loaded directly from the model path (without numerical version directories).

Versioning is optional starting from 2026.3 OVMS release.

When a model (or a new model version in versioned layout) is detected, the server loads the model files and starts serving it. This operation might fail for the following reasons :
- There is a problem with accessing model files (due to network connectivity issues to the remote storage or insufficient permissions).
- Model files are malformed and can not be imported by the OpenVINO&trade; Runtime.
- Model requires a custom CPU extension.


Below is an example structure with one invalid version and one valid non-versioned model:
```
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── somefile.bin
│       └── anotherfile.txt
└── model2
    ├── ir_model.bin
    ├── ir_model.xml
    └── mapping_config.json
```
- In the above example, the server will detect only Directory `1` of `model1`. It will not detect Directory `2` as a valid model version because it does not contain valid OpenVINO model files.

- `model2` is a valid non-versioned layout and can be loaded directly from model files in the model directory.

- The root cause is reported in the server logs or the response from a call to GetModelStatus function.

- A model (or model version) that is detected but not loaded will not be served. It will report status `LOADING` with the error message: `Error occurred while loading version`.

- When model files become accessible or fixed, the server will try to load them again on the next version update attempt.

- Model import will fail if the OVMS process does not have read permissions to model files and list permissions on the model folder (and version subfolder when versioned layout is used).


## Python Support Issues

Python support is optional in Model Server. The following issues may occur when using Python features (Python nodes, LLM models with Jinja2 templates):

### Python Runtime Initialization Fails (Server Continues with Python Features Disabled)
- **Cause**: System Python library (`libpython.so`) is not found
- **Symptoms**: Server starts, but Python features are unavailable; logs mention Python runtime initialization failure and fallback
- **Resolution**:
  - Verify you deployed a with-Python package, not without-Python package
  - Install system Python development libraries:
    - **Ubuntu/Debian**: `sudo apt install libpython3.12-dev` (adjust version as needed)
    - **RHEL/CentOS**: `sudo yum install python312-devel`
  - Verify: `find /usr -name "libpython*.so*" -type f`

### Python Backend Creation Fails (Server Continues with Python Features Disabled)
- **Cause**: Python module `pyovms` cannot be found or imported
- **Symptoms**: Server starts, but Python features are unavailable; logs mention missing or broken `pyovms` module
- **Resolution**:
  - Set `PYTHONPATH` to include the Python libraries directory from OVMS package (for example, prepend `${OVMS_PACKAGE_PATH}/lib/python` to `PYTHONPATH`).
  - Verify: `python3 -c "import pyovms; print(pyovms.__file__)"`
  - Check that Python dependencies are installed: `pip3 install Jinja2==3.1.6 MarkupSafe==3.0.2`
  - If using Python nodes: `pip3 install numpy`

### Warning: "Python calculators plugin failed to load" but Server Continues
- **Cause**: Library `libpython_calculators.so` not found in library search path
- **Symptoms**: Server starts successfully, but Python nodes cannot be loaded
- **Impact**: Non-Python models work normally; Python node graphs return error when loaded
- **Resolution** (if you need Python nodes):
  - Verify file exists: `find ${OVMS_LIB_PATH} -name "libpython_calculators.so"`
  - Check library path: `echo $LD_LIBRARY_PATH` (should include `${OVMS_LIB_PATH}`)
  - Update search path: `export LD_LIBRARY_PATH=${OVMS_LIB_PATH}:$LD_LIBRARY_PATH`
- **Resolution** (if you don't need Python nodes):
  - No action required; graceful degradation allows non-Python models to work

### Python Node Graph Returns Error When Loaded
- **Error**: "PythonExecutorCalculator not found" or "Python calculators plugin not available"
- **Cause**: Python calculators plugin failed to load (see above) or Python support was compiled out (`PYTHON_DISABLE=1`)
- **Resolution**:
  - Follow "Python calculators plugin failed to load" steps above
  - Or use without-Python package if you don't need Python nodes

### LLM Models with Complex Jinja2 Templates Don't Render Correctly
- **Symptoms**: Chat template output is missing parts or shows literal template syntax
- **Cause**: Without-Python package in use; complex template features require Python
- **Resolution**:
  - Use with-Python package instead of without-Python package
  - Follow Python setup steps above (PYTHONPATH, dependencies)

### Verify Python Support Status
- OVMS loads Python runtime dynamically, so `ldd ${OVMS_BIN}` may not list Python-related libraries; instead verify `${OVMS_LIB_PATH}/libovmspython.*` is present and check logs for `Python runtime library loaded successfully` / `Python calculators plugin loaded successfully`.
- Check server logs for `KFS Python tensor bridge activated` or `Python calculators plugin libpython_calculators.so failed to load`.
- Submit a Python node graph request; when Python support is unavailable, the request fails gracefully with a clear error.
- Confirm OVMS Python package files exist (for with-Python package), for example under `${OVMS_PACKAGE_PATH}/lib/python`.

## Client Request Issues
- When the model server starts successfully and all the models are imported, there could be a couple of reasons for errors in the request handling.
- The information about the failure reason is passed to the client in the response. It is also logged on the model server in the DEBUG mode.

The possible issues could be :
* Incorrect shape of the input data.
* Incorrect input key name which does not match the tensor name or set input key name in `mapping_config.json`.
* Incorrectly serialized data on the client-side.

## Resource Allocation
- RAM consumption might depend on the size and volume of the models configured for serving. It should be measured experimentally, however it can be estimated that each model will consume RAM size equal to the size of the model weights file (.bin file).

- Every version of the model enabled in the version policy creates a separate OpenVINO&trade; Runtime `ov::Model` and `ov::CompiledModel` object. By default, only the latest version is enabled.

- OpenVINO&trade; model server consumes all available CPU resources unless they are restricted by the operating system, Docker or Kubernetes capabilities.

*Note* When insufficient memory is allocated to the container, it might get terminated by the Docker engine [OOM Killer](https://docs.docker.com/config/containers/resource_constraints/). There will be no termination root cause
mentioned in the OVMS logs but such a situation can be confirmed by `docker inspect <terminated_container>` and reported State `"OOMKilled": true`.
It will be also included in the host system logs like `Memory cgroup out of memory: Killed process`.


## Usage Monitoring
- Prometheus standard metrics are available via the REST /metrics endpoint.
- It is possible to track the usage of the models including processing time while DEBUG mode is enabled.
- With this setting model server logs will store information about all the incoming requests.
- You can parse the logs to analyze: the volume of requests, processing statistics, and most used models.

## Configuring S3 Storage For Use With a Proxy
- To use S3 storage behind a proxy, an environment variable should be configured. The S3 loader module is using the following format
```
http://user:password@hostname:port
or
https://user:password@hostname:port
```
where user and password are optional. The OVMS will try to use the following environment variables :
```
https_proxy
HTTPS_PROXY
http_proxy
HTTP_proxy
```

> **Note**: that neither `no_proxy` or `NO_PROXY` is used.

## Using GCS Model Behind a Proxy

- If your environment is required to use proxy but `http_proxy`/`https_proxy` is not passed to the server container there will be 15 minutes timeout when accessing GCS models.
- During that time no logs will be captured by OVMS. Currently, there is no option to change the timeout duration for GCS.

## Model Cache Issues

- Cache folder (by default `/opt/cache` or defined by `--cache_dir`) should be mounted into docker container with read-write access. Unless changed by the docker run command, the model server has a security context of ovms account with uid 5000.
- The biggest speedup in the model loading time is expected for GPU device. For CPU device the gain will depend on the model topology. In some rare cases, it is possible the load time will not be improved noticeably or it might be even slightly slower.
