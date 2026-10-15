# OVMS Docker Build & Transfer Script

Automated script to build OpenVINO Model Server Docker images and transfer them to remote servers.

## Quick Start

### Basic Usage: Build and Transfer to ov-arls-38

```bash
./build_and_transfer_docker.sh -t v1 --load
```

This will:
1. Build the default OVMS image with tag `v1`
2. Create a compressed archive
3. Transfer to `ov-arls-38`
4. Load the image on the remote server

### Build with GPU Support

```bash
./build_and_transfer_docker.sh -t v1-gpu --gpu --load
```

### Build on Ubuntu 22 (instead of default Ubuntu 24)

```bash
./build_and_transfer_docker.sh -t myversion -o ubuntu22 --load
```

### Use Custom SSH User

```bash
./build_and_transfer_docker.sh -t v1 -u myuser --load
```

### Build, Load, and Push to Docker Registry

```bash
./build_and_transfer_docker.sh -t latest --load -r docker.io/myuser
```

This tags the image as `docker.io/myuser/openvino/model_server:latest` and pushes it.

## Options

```
-h, --host HOST              Remote host (default: ov-arls-38)
-u, --user USER              Remote SSH user (default: current user)
-d, --dir DIR                Remote transfer directory (default: /tmp)
-t, --tag TAG                Docker image tag (default: latest)
-n, --name NAME              Docker image name (default: openvino/model_server)
-o, --os OS                  Base OS (default: ubuntu24)
                             Options: ubuntu24, ubuntu22, redhat
-g, --gpu                    Build GPU-enabled image
-l, --load                   Load image on remote server after transfer
-r, --registry REGISTRY      Push to registry after loading (e.g., docker.io/user)
-j, --jobs JOBS              Number of parallel build jobs (default: auto)
--no-cache                   Build without Docker cache
--help                       Show help message
```

## Environment Variables (Optional)

Instead of command-line flags, you can also use environment variables:

```bash
export REMOTE_HOST=ov-arls-38
export REMOTE_USER=myuser
export BASE_OS=ubuntu24
export BUILD_GPU=0

./build_and_transfer_docker.sh -t myversion --load
```

## Common Scenarios

### Scenario 1: Build for testing with Voxtral changes

```bash
# Build CPU image with WebRTC voxtral support
./build_and_transfer_docker.sh -t voxtral-test --load
```

Then on the remote server:
```bash
ssh ov-arls-38

docker run -it \
  -e OVMS_WEBRTC_VOXTRAL_MODEL_PATH=/models/voxtral \
  -v /models:/models:ro \
  -p 19112:9112 \
  openvino/model_server:voxtral-test
```

### Scenario 2: Build GPU image with all optimizations

```bash
./build_and_transfer_docker.sh \
  -t gpu-optimized \
  --gpu \
  -o ubuntu24 \
  -j 16 \
  --load \
  -r docker.io/myuser
```

### Scenario 3: Quick rebuild with no cache

```bash
./build_and_transfer_docker.sh -t latest --no-cache --load
```

## What the Script Does

1. **Builds** OVMS Docker image using `make docker_build`
2. **Compresses** the image to .tar.gz (typically 500MB-2GB depending on options)
3. **Transfers** via SCP to remote server
4. **Optionally loads** the image into Docker on the remote
5. **Optionally pushes** to a Docker registry

## Transfer Details

- **Local archive location**: `./ovms-{tag}-{timestamp}.tar.gz`
- **Remote default location**: `/tmp/ovms-{tag}-{timestamp}.tar.gz`
- **Custom remote location**: Use `-d` flag
- **SSH method**: Direct SCP with passwordless SSH (set up SSH keys first)

## Requirements

- Local machine: Docker, make, git
- Remote server: Docker, SSH access, disk space (~3-5GB for transfer dir)
- Network: SCP/SSH connectivity between machines
- Permissions: Sudo access not required if using non-root Docker

## Troubleshooting

### SSH Connection Failed

```bash
# Test SSH connection first
ssh -v ov-arls-38

# If you need password auth (not recommended), use:
ssh -u myuser ov-arls-38

# Set up SSH keys (one-time):
ssh-copy-id -i ~/.ssh/id_rsa.pub user@ov-arls-38
```

### Docker Build Failed

```bash
# Check Docker is running
docker ps

# Check available disk space (need ~10-20GB)
df -h

# Try building with more resources
./build_and_transfer_docker.sh -t test -j 1 --load
```

### Transfer/Load Failed on Remote

```bash
# SSH to remote and check Docker
ssh ov-arls-38 docker ps

# Check disk space on remote
ssh ov-arls-38 df -h

# Manually check if archive exists
ssh ov-arls-38 ls -lh /tmp/ovms-*.tar.gz
```

### Image Not Loading on Remote

```bash
# SSH to remote and try manual load
ssh ov-arls-38
cd /tmp
gunzip -c ovms-v1-*.tar.gz | docker load
docker images | grep openvino
```

## Performance Tips

1. **Use parallel jobs**: `-j 32` (or number of cores available)
2. **Skip cache rebuild**: Only use `--no-cache` when dependencies change
3. **Network**: Transfer time depends on archive size and network speed
4. **Disk space**: Keep enough space in `/tmp` on both local and remote

## Advanced: Building Different Variants

### Text-to-Speech focus (Kokoro)
```bash
./build_and_transfer_docker.sh -t tts-latest --load
```

### Speech-to-Text focus (Voxtral)
```bash
./build_and_transfer_docker.sh -t stt-latest --load
```

### MultiModal focus (Omni)
```bash
./build_and_transfer_docker.sh -t multimodal-latest --gpu --load
```

## Post-Transfer: Running the Image

On the remote server:

```bash
# Check available images
docker images | grep openvino

# Run with Voxtral WebRTC
docker run -d \
  --name ovms-voxtral \
  -e OVMS_WEBRTC_VOXTRAL_MODEL_PATH=/models/voxtral-realtime-2602 \
  -e OVMS_WEBRTC_STT_DEVICE=GPU \
  -v /models:/models:ro \
  -p 9112:9112 \
  -p 8000:8000 \
  openvino/model_server:v1

# Check logs
docker logs -f ovms-voxtral

# Stop when done
docker stop ovms-voxtral
docker rm ovms-voxtral
```

## Cleanup

```bash
# Clean up local archives (keep last N versions)
rm -f ovms-*.tar.gz

# On remote: clean up archives
ssh ov-arls-38 rm -f /tmp/ovms-*.tar.gz

# Remove old Docker images (keep latest)
docker image prune -f
```

## Script Output Example

```
[INFO] ==========================================
[INFO] OVMS Docker Build & Transfer Configuration
[INFO] ==========================================
[INFO] Remote Host: ov-arls-38
[INFO] Remote User: user
[INFO] Transfer Dir: /tmp
[INFO] Image: openvino/model_server:v1
[INFO] Base OS: ubuntu24
[INFO] Build Type: CPU
[INFO] Load on Remote: YES
[INFO] Archive: ovms-v1-20261013-143022.tar.gz

[INFO] ==========================================
[INFO] Step 1: Building Docker Image
[INFO] ==========================================
[INFO] Executing: make docker_build BASE_OS=ubuntu24 OVMS_CPP_IMAGE_TAG=v1

... [build output] ...

[SUCCESS] Docker image built successfully: openvino/model_server:v1
[INFO] ==========================================
[INFO] Step 2: Saving Docker Image to Archive
[INFO] ==========================================
[SUCCESS] Image saved to: /home/user/model_server/ovms-v1-20261013-143022.tar.gz (Size: 850M)

[INFO] ==========================================
[INFO] Step 3: Transferring to Remote Server
[INFO] ==========================================
[SUCCESS] Image transferred successfully to: ov-arls-38:/tmp/ovms-v1-20261013-143022.tar.gz

[INFO] ==========================================
[INFO] Step 4: Loading Image on Remote Server
[INFO] ==========================================
[SUCCESS] Image loaded on remote server

[INFO] ==========================================
[INFO] Summary
[INFO] ==========================================
[SUCCESS] All steps completed successfully!
```

## License

This script is part of OpenVINO Model Server and follows the same Apache 2.0 license.
