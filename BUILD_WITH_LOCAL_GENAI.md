# Building OVMS with Local OpenVINO GenAI Sibling Branch

This guide explains how to build OpenVINO Model Server (OVMS) using a custom build of OpenVINO GenAI from a sibling repository branch.

## Problem

When building OVMS locally, you may encounter compilation errors like:
```
error: 'class ov::genai::ASRPipeline' has no member named 'create_stream'
```

This happens because the pre-built OpenVINO GenAI binary packages available for download don't include the latest features from your development branch.

## Solution

Use the `OV_GENAI_LOCAL_PATH` build argument to mount your local OpenVINO GenAI repository into the Docker build context.

## Prerequisites

1. **Clone OpenVINO GenAI sibling repository** locally:
   ```bash
   cd ~/
   git clone https://github.com/openvinotoolkit/openvino.genai.git
   # OR if you have a custom branch:
   git clone --branch your-branch https://github.com/yourfork/openvino.genai.git
   ```

2. **Ensure OVMS repository is also locally available**:
   ```bash
   cd ~/model_server
   ```

## Building with Local OpenVINO GenAI

### Option 1: Using `make docker_build` directly

Set the `OV_GENAI_LOCAL_PATH` environment variable or pass it as a build argument:

```bash
cd ~/model_server

# Using environment variable
export OV_GENAI_LOCAL_PATH=~/openvino.genai
make docker_build BASE_OS=ubuntu24 OV_GENAI_LOCAL_PATH=${PWD}/../openvino.genai

# OR pass it as a Makefile variable
make docker_build BASE_OS=ubuntu24 OV_GENAI_LOCAL_PATH=$(realpath ../openvino.genai)
```

### Option 2: Using `build_and_transfer_docker.sh`

The build script can be enhanced to support this. Create a wrapper:

```bash
#!/bin/bash
# build_with_local_genai.sh

export OV_GENAI_LOCAL_PATH=$(realpath ../openvino.genai)
export ov_use_binary=0  # Force source build

cd ~/model_server
./build_and_transfer_docker.sh "$@"
```

Make it executable and run:
```bash
chmod +x build_with_local_genai.sh
./build_with_local_genai.sh --tag voxtral-local --load
```

### Option 3: Direct Docker Build

For more control, build directly with Docker:

```bash
cd ~/model_server

# CPU build with local GenAI
docker build \
  -f Dockerfile.ubuntu \
  --build-arg ov_use_binary=0 \
  --build-arg OV_GENAI_LOCAL_PATH=$(realpath ../openvino.genai) \
  -t openvino/model_server:voxtral-local \
  .

# GPU build with local GenAI
docker build \
  -f Dockerfile.ubuntu \
  --build-arg ov_use_binary=0 \
  --build-arg GPU=1 \
  --build-arg OV_GENAI_LOCAL_PATH=$(realpath ../openvino.genai) \
  -t openvino/model_server-gpu:voxtral-local \
  .
```

## How It Works

The Dockerfiles now include support for mounting the local openvino.genai repository:

### For Ubuntu (Dockerfile.ubuntu) - Lines 240-250:
```dockerfile
ARG OV_GENAI_LOCAL_PATH=""
RUN --mount=type=bind,source=${OV_GENAI_LOCAL_PATH},target=/tmp/ov_genai_bind,required=false \
    if [ "$ov_use_binary" == "0" ]; then \
        if [ -n "$OV_GENAI_LOCAL_PATH" ] && [ -d /tmp/ov_genai_bind ]; then \
            cp -r /tmp/ov_genai_bind /openvino_genai && ... ; \
        else \
            git clone https://github.com/$ov_genai_org/openvino.genai ... ; \
        fi \
    fi
```

**Key points:**
1. `--mount=type=bind` - Read-only mount of host directory into container
2. `required=false` - Build succeeds even if mount fails (gracefully falls back to GitHub clone)
3. Conditional logic: If path provided and exists, copy it; otherwise clone from GitHub
4. All submodules are initialized regardless of source

### For RedHat (Dockerfile.redhat) - Similar logic

## Environment Variables

The following environment variables can control the build:

| Variable | Value | Effect |
|----------|-------|--------|
| `ov_use_binary` | `0` | Force source build (required for custom GenAI) |
| `ov_genai_branch` | `branch-name` | GitHub branch to clone (only if not using local) |
| `ov_genai_org` | `username` | GitHub org/user to clone from (default: openvinotoolkit) |
| `OV_GENAI_LOCAL_PATH` | `/path/to/genai` | Local repository path (overrides GitHub) |

## Example Workflows

### Workflow 1: Quick Iteration with Local GenAI

```bash
# Terminal 1: Start the build
cd ~/model_server
make docker_build BASE_OS=ubuntu24 ov_use_binary=0 OV_GENAI_LOCAL_PATH=../openvino.genai

# Terminal 2 (while building): Monitor container
docker ps | grep -i build
docker logs -f <container_id>

# After build completes, transfer to remote
./build_and_transfer_docker.sh --tag dev-$(date +%s) --load
```

### Workflow 2: Test Custom GenAI Feature

```bash
# Clone both repos
git clone https://github.com/yourfork/openvino.genai --branch your-feature
cd openvino.genai
# Make changes to GenAI...
git commit -m "Add create_stream method"

cd ../model_server
# Build OVMS with your GenAI changes
make docker_build ov_use_binary=0 OV_GENAI_LOCAL_PATH=../openvino.genai

# Test locally or transfer
docker run -it openvino/model_server:latest bash
```

### Workflow 3: Build, Load, and Test on Remote

```bash
# Build with local GenAI
docker build -f Dockerfile.ubuntu \
  --build-arg ov_use_binary=0 \
  --build-arg OV_GENAI_LOCAL_PATH=$(realpath ../openvino.genai) \
  -t openvino/model_server:genai-test \
  .

# Save and transfer
docker save openvino/model_server:genai-test | gzip > ovms-genai-test.tar.gz
scp ovms-genai-test.tar.gz ov-arls-38:/tmp/

# Load on remote
ssh ov-arls-38 "cd /tmp && gunzip -c ovms-genai-test.tar.gz | docker load"

# Run on remote
ssh ov-arls-38 "docker run -it openvino/model_server:genai-test /bin/bash"
```

## Troubleshooting

### Issue: "No such file or directory: openvino.genai"

**Cause:** The path doesn't exist or is relative and not resolved correctly.

**Fix:**
```bash
# Use absolute path
export OV_GENAI_LOCAL_PATH=$(realpath ~/openvino.genai)
make docker_build OV_GENAI_LOCAL_PATH=${OV_GENAI_LOCAL_PATH}
```

### Issue: Build uses GitHub instead of local repo

**Cause:** `ov_use_binary=1` (default) or `OV_GENAI_LOCAL_PATH` not set.

**Fix:**
```bash
# Ensure source build mode
make docker_build ov_use_binary=0 OV_GENAI_LOCAL_PATH=../openvino.genai
```

### Issue: "Git submodule update fails"

**Cause:** Local GenAI repo has detached HEAD or incomplete submodules.

**Fix:**
```bash
cd ~/openvino.genai
git checkout your-branch
git pull
git submodule update --init --recursive
cd ~/model_server
make docker_build ov_use_binary=0 OV_GENAI_LOCAL_PATH=../openvino.genai
```

### Issue: Build still takes long time

**Cause:** Docker is re-building GenAI from scratch every time.

**Fix:** Use `--build-arg GPU=0 --parallel-jobs 16` to speed up:
```bash
make docker_build ov_use_binary=0 OV_GENAI_LOCAL_PATH=../openvino.genai JOBS=16
```

## Build Times

Approximate build times when compiling GenAI from source:

| Configuration | Time | Notes |
|--------------|------|-------|
| CPU only | 40-60 min | Depends on CPU cores |
| GPU support | 50-70 min | More dependencies |
| With `JOBS=32` | 30-40 min | Parallelism helps significantly |
| Cached (2nd run) | 20-30 min | Reuses layers |

## Verifying the Build

After the Docker image is built, verify it includes your custom GenAI:

```bash
# Check that create_stream is available (example for ASRPipeline)
docker run openvino/model_server:voxtral-local bash -c \
  'strings /opt/intel/openvino/runtime/lib/intel64/libopenvino_genai.so | grep create_stream'

# Should output if method exists in binary
```

## Integration with CI/CD

For GitHub Actions or Jenkins, use secrets to provide SSH keys:

```yaml
# .github/workflows/build-ovms.yml
- name: Build OVMS with local GenAI
  env:
    OV_GENAI_LOCAL_PATH: ${{ github.workspace }}/openvino.genai
  run: |
    make docker_build ov_use_binary=0 OV_GENAI_LOCAL_PATH=${OV_GENAI_LOCAL_PATH}
```

## Additional Resources

- [OpenVINO GenAI Repository](https://github.com/openvinotoolkit/openvino.genai)
- [OpenVINO Model Server Documentation](https://docs.openvino.ai/latest/model-server.html)
- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)

---

**Last Updated:** 2026-10-13  
**OVMS Version:** 2026.5+  
**Docker Support:** BuildKit (required for `--mount` feature)
