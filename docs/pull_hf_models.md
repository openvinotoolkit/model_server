# OVMS Pull mode {#ovms_docs_pull}

This document describes how to leverage OpenVINO Model Server (OVMS) pull feature to automate deployment configuration with Generative AI models. When pulling from [Hugging Face Hub](https://huggingface.co/) or when pulling GGUF model, no additional steps are required. However, when pulling models in Pytorch format, you have to install additional python dependencies when using baremetal execution so that optimum-cli is available for ovms executable or rely on the docker image `openvino/model_server:latest-py`. In summary you have 3 options:

- pulling pre-configured models in IR format (described below)
- pulling GGUF models from Hugging Face
- pulling models with automatic conversion and quantization via optimum-cli. Described in the [pulling with conversion](https://github.com/openvinotoolkit/model_server/blob/releases/2026/2/docs/pull_optimum_cli.md)

> **Note:** Models in IR format must be exported using `optimum-cli` including tokenizer and detokenizer files also in IR format, if applicable. If missing, tokenizer and detokenizer should be added using `convert_tokenizer --with-detokenizer` tool.

## Pulling pre-configured models

There is a special OVMS mode to pull the model from Hugging Face without starting the service. It is triggered by `--pull` parameter. The application quits after the model is downloaded. Without `--pull` option, the model will be deployed and server started.

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed
```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:weekly --pull --source_model <model_name_in_HF> --model_repository_path /models --model_name <external_model_name> --target_device <DEVICE> [--gguf_filename SPECIFIC_QUANTIZATION_FILENAME.gguf] --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --pull --source_model <model_name_in_HF> --model_repository_path <model_repository_path> --model_name <external_model_name> --target_device <DEVICE> [--gguf_filename SPECIFIC_QUANTIZATION_FILENAME.gguf] --task <task> [TASK_SPECIFIC_PARAMETERS]
```
:::
::::

*Note:* GGUF format model is only supported with `--task text_generation`. For list of supported models check [blog](https://blog.openvino.ai/blog-posts/openvino-genai-supports-gguf-models).

Example for pulling `OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov`:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:latest --pull --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.

```text
ovms --pull --source_model "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov" --model_repository_path /models --model_name Phi-3-mini-FastDraft-50M-int8-ov --task text_generation
```
:::
::::

Example for pulling GGUF model `unsloth/Llama-3.2-1B-Instruct-GGUF` with Q4_K_M quantization on baremetal host:

::::{tab-set}
:::{tab-item} With Docker
:sync: docker
**Required:** Docker Engine installed

```text
docker run $(id -u):$(id -g) --rm -v <model_repository_path>:/models:rw openvino/model_server:weekly --pull --source_model "unsloth/Llama-3.2-1B-Instruct-GGUF" --model_repository_path /models --model_name unsloth/Llama-3.2-1B-Instruct-GGUF --task text_generation --gguf_filename Llama-3.2-1B-Instruct-Q4_K_M.gguf
```
:::

:::{tab-item} On Baremetal Host
:sync: baremetal
**Required:** OpenVINO Model Server package - see [deployment instructions](./deploying_server_baremetal.md) for details.
```text
ovms --pull --source_model "unsloth/Llama-3.2-1B-Instruct-GGUF" --model_repository_path /models --model_name unsloth/Llama-3.2-1B-Instruct-GGUF --task text_generation --gguf_filename Llama-3.2-1B-Instruct-Q4_K_M.gguf
```
:::
::::

## Pulling models outside OpenVINO organization

It is possible to pull models outside of OpenVINO organization. 

Example for pulling `Echo9Zulu/phi-4-int4_asym-awq-ov`:

```text
ovms --pull --source_model Echo9Zulu/phi-4-int4_asym-awq-ov --model_repository_path /models --model_name phi-4-int4_asym-awq-ov --target_device CPU --task text_generation 
```

> **Note:** These models aren't tested properly and their accuracy or performance may be low.

Check [parameters page](./parameters.md) for detailed descriptions of configuration options and parameter usage.

In case you want to setup model and start server in one step, follow [instructions](./starting_server.md).

> **Note:**  When using pull mode you need both read and write access rights to models repository.

## Pulling Image Generation Models with LoRA Adapters

For image generation tasks, you can additionally specify LoRA adapters to be downloaded alongside the base model using the `--source_loras` parameter:

```text
ovms --rest_port 8000 \
  --model_repository_path /models/ \
  --task image_generation \
  --source_model stabilityai/stable-diffusion-xl-base-1.0 \
  --source_loras "xray=DoctorDiffusion/doctor-diffusion-s-xray-xl-lora@DD-xray-v1.safetensors,ukiyo=KappaNeuro/ukiyo-e-art@Ukiyo-e Art.safetensors"
```

The `--source_loras` format is a comma-separated list of `alias=source[:alpha]` entries. Supported source types:
- HuggingFace repository: `alias=org/repo` or `alias=org/repo@filename.safetensors`
- Direct URL: `alias=https://url/to/file.safetensors`
- Local file (Linux): `alias=/path/to/file.safetensors`
- Local file (Windows): `alias=C:\path\to\file.safetensors`
- Relative local file: `alias=./path/to/file.safetensors`

Each adapter can optionally specify a default alpha weight: `alias=source:0.7` (default: `1.0`).

For more details, see the [LoRA Adapters documentation](./image_generation/reference.md#lora-adapters).

## Resuming an interrupted pull

Pulling Generative AI models from Hugging Face often involves transferring multi-gigabyte LFS files (e.g. `openvino_model.bin`). To make this robust against network errors and operator interventions, OVMS pull mode persists the in-progress download state on disk and resumes from where it stopped on the next `--pull` invocation. No extra flags are required — simply re-run the same `--pull` command against the same `--model_repository_path` and OVMS will continue any partially downloaded LFS files instead of starting from scratch.

### What is persisted

While a pull is in flight, OVMS / libgit2 keeps the following on disk under your `--model_repository_path`:

| Artifact | Purpose |
|---|---|
| `<repo>.lfswip` (sibling of the repository directory) | Marker file indicating that an LFS download is work-in-progress. |
| `<file>.lfs_part` (next to each LFS-tracked file) | Partially downloaded LFS object. The next `--pull` resumes the HTTP transfer from the existing byte offset. |
| LFS pointer file (in place of the final binary) | Standard `version https://git-lfs.github.com/spec/v1` pointer that allows libgit2 to identify which OID still needs to be fetched. |

When the LFS transfer for a file completes successfully, the `.lfs_part` file is renamed to its final name and the pointer is replaced. Once **all** LFS files are present, the `.lfswip` marker is removed and the repository is considered clean.

### Resume after Ctrl+C / SIGINT (graceful cancel)

Pressing **Ctrl+C** (or sending `SIGINT` / `SIGTERM` on Linux, `CTRL_BREAK_EVENT` on Windows) while `ovms --pull` is running triggers a graceful cancellation:

1. OVMS marks the server as shutting down.
2. libgit2 clone / LFS callbacks observe the cancellation request and abort the in-flight HTTP transfer cleanly.
3. The process exits with a non-zero status code. Partial `.lfs_part` files and the `.lfswip` marker are left on disk on purpose.
4. Re-running the **same** `--pull` command resumes each partial file using HTTP `Range` requests and finishes the remaining downloads.

This is the recommended way to interrupt a pull — it avoids corrupted partial data and lets you resume without re-downloading completed files.

### Resume after process termination (forced kill / power loss)

If the OVMS process is killed forcibly (`SIGKILL`, OOM killer, container stop with no grace period, host crash, power loss), the on-disk state is the same as for a graceful cancel: any LFS files that were in flight remain as `<file>.lfs_part` plus an LFS pointer file, and the `.lfswip` marker is still present. The next `--pull` invocation:

1. Detects the `.lfswip` marker and the leftover LFS pointer files.
2. For each affected file, opens an HTTP `Range` request starting at the current size of the corresponding `.lfs_part` file and continues the transfer.
3. Cleans up the marker once every LFS file is fully present.

If a forced termination corrupted an in-progress write, the resumed transfer will detect the size/hash mismatch on completion and the file will be re-downloaded on a subsequent attempt. User-edited or user-deleted files are **not** restored automatically — once a `--pull` has finished successfully, OVMS treats the local repository as authoritative and will not overwrite or re-fetch files that you have modified or removed. To force OVMS to re-download a model from scratch, pass `--overwrite_models` on the next `--pull` invocation; the existing model directory under `--model_repository_path` will be replaced with a fresh download.

### Tuning resume behavior

The number of resume attempts per LFS file and the interval between them can be tuned via environment variables read once on process start (defaults shown):

| Environment variable | Default | Description |
|---|---|---|
| `GIT_LFS_RESUME_ATTEMPTS` | `5` | Maximum number of resume attempts for a single LFS file before giving up. `0` disables resume. |
| `GIT_LFS_RESUME_INTERVAL_SECONDS` | `10` | Delay between consecutive resume attempts. |

On startup OVMS logs the resolved configuration, e.g.:

```text
[INFO] LFS resume: attempts=5 interval=10 s
```

> **Note:** Resume relies on the remote server honoring HTTP `Range` requests. Hugging Face Hub supports this by default; private mirrors must allow ranged GETs for resume to work.
