# OVMS Pull mode for CNCF ModelPack images {#ovms_docs_pull_oci}

Besides Hugging Face Hub, OVMS can pull models that are distributed as OCI artifacts following the [CNCF ModelPack specification](https://github.com/modelpack/model-spec). Because ModelPack images are ordinary OCI artifacts, they can be stored in and served from any OCI registry — Docker Hub, GHCR, quay, Artifactory or a self-hosted registry — with the same tooling, authentication and mirroring you already use for container images.

A model is requested by prefixing `--source_model` with the `oci://` scheme:

```text
ovms --pull --source_model oci://ghcr.io/<org>/<model>:<tag> --model_repository_path /models --task text_generation
```

> **Note:** The `oci://` scheme is required. A bare `registry/name:tag` string is indistinguishable from a Hugging Face repository id (`org/model`), so OVMS never guesses.

## Prerequisites

OCI pulling is delegated to [`llmman`](https://github.com/llmmanorg/llmman), which implements registry authentication, the ModelPack media types, resumable blob downloads and a local content-addressed store. Install it and make sure it is on `PATH`:

```text
curl -fsSL https://llmmanorg.github.io/install.sh | sh
```

OVMS reports an error when `llmman` is not available on `PATH`.

Public registries can be used without credentials. For a private registry, authenticate with `llmman login <registry>` and make the resulting Docker-compatible credential configuration available to the user running OVMS. `llmman` reads credentials from the Docker credential configuration; it does not use `HF_TOKEN` for OCI registry authentication.

In a container, logging in on the host alone is not sufficient because the container does not automatically see the host's credentials. Mount the Docker config directory read-only and point `DOCKER_CONFIG` at it. For example:

```text
docker run --mount type=bind,src="$HOME/.docker",dst=/home/ovms/.docker,readonly -e DOCKER_CONFIG=/home/ovms/.docker ... openvino/model_server:latest
```

If the Docker config uses a credential helper (`credsStore` or `credHelpers`), that helper and any service it depends on must also be available inside the container. Alternatively, configure credentials in a mounted Docker config that does not rely on an unavailable helper. Do not bake registry credentials into the image.

In Kubernetes, store the Docker config in a Secret and mount it into the OVMS pod. For example, create a Secret from a Docker config file:

```text
kubectl create secret generic registry-auth --from-file=config.json="$HOME/.docker/config.json"
```

Mount the Secret read-only and set `DOCKER_CONFIG` to the mount directory in the OVMS container spec:

```yaml
spec:
  containers:
    - name: ovms
      image: openvino/model_server:latest
      env:
        - name: DOCKER_CONFIG
          value: /var/run/registry-auth
      volumeMounts:
        - name: registry-auth
          mountPath: /var/run/registry-auth
          readOnly: true
  volumes:
    - name: registry-auth
      secret:
        secretName: registry-auth
```

The Secret must contain a `config.json` usable by llmman. As with a Docker bind mount, any credential helper it references must also be installed and usable in the pod.

## Supported payloads

`llmman resolve` reports the format of the layers in the image, and OVMS reacts to it:

| ModelPack payload | What OVMS does | Extra requirements |
|---|---|---|
| OpenVINO IR (`openvino_model.xml` + `.bin`) | Serves it directly out of the `llmman` store; only `graph.pbtxt` is written to the model repository. `llmman resolve` only reports `gguf` and `safetensors`, so an IR-only image fails with `OCI_LLMMAN_RESOLVE_FAILED` for now. IR is used in place only when a `safetensors` image also carries IR files. | none |
| GGUF | Serves the `.gguf` file directly out of the `llmman` store. | `--task text_generation` only |
| Hugging Face safetensors | Converts to OpenVINO IR with `optimum-cli` into the model repository, honoring `--weight-format` and `--extra_quantization_params`. | Python dependencies, see [pulling with conversion](./pull_optimum_cli.md) |

## Examples

Pull a GGUF ModelPack image without starting the server:

```text
ovms --pull --source_model oci://docker.io/ai/qwen3.5:0.8b --model_repository_path /models --task text_generation
```

Pull and start in one step, overriding the served model name:

```text
ovms --rest_port 8000 --source_model oci://docker.io/ai/qwen3.5:0.8b --model_repository_path /models --model_name qwen --task text_generation
```

Pull a safetensors ModelPack image and quantize it during the conversion:

```text
ovms --pull --source_model oci://ghcr.io/my-org/qwen3-8b:1.0 --model_repository_path /models --task text_generation --weight-format int4
```

## Naming and on-disk layout

The `oci://` scheme is dropped from the served model name, so the reference you typed is what clients use in the `model` field:

```text
--source_model oci://ghcr.io/my-org/model:1.0   ->   model name "ghcr.io/my-org/model:1.0"
```

Pass `--model_name` to override it.

Inside `--model_repository_path` the reference additionally has `:` replaced with `+`, because `:` is not a legal filename character on Windows:

```text
<model_repository_path>/ghcr.io/my-org/model+1.0/graph.pbtxt
```

For OpenVINO IR and GGUF payloads that directory holds only `graph.pbtxt` — the weights stay in `llmman`'s content-addressed store and are referenced by absolute path, so pulling the same image for several servables does not duplicate them on disk. Removing a model therefore takes two steps: delete the directory from the model repository, and reclaim the blobs with `llmman rm <reference>`.

## Limitations

- `--task` must be provided explicitly. The task is normally inferred by reading `config.json` from the model source, which for a registry reference would mean pulling the whole image before the command line has even been parsed.
- `--gguf_filename` is rejected. The layer media types in a ModelPack image already identify the payload, so there is nothing to select.
- Speculative decoding (`--draft_source_model`) and image generation LoRA adapters (`--source_loras`) are still resolved from Hugging Face, also when the base model comes from a registry.

Check the [parameters page](./parameters.md) for detailed descriptions of configuration options.
