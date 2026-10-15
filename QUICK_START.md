# OVMS Docker Build & Transfer - Quick Reference

## 📦 Scripts Created

1. **`build_and_transfer_docker.sh`** (9.3 KB)
   - Main script: Builds OVMS Docker image and transfers to remote server
   - Full control with many options
   - Supports CPU/GPU, multiple OS variants, registry push

2. **`build_profiles.sh`** (2.5 KB)  
   - Convenience wrapper with predefined build profiles
   - Quick shortcuts for common scenarios

3. **`BUILD_AND_TRANSFER_README.md`** (7.4 KB)
   - Comprehensive documentation
   - Troubleshooting guide
   - Advanced usage examples

## 🚀 Quick Start (Copy & Paste)

### 1️⃣ Build & Transfer to ov-arls-38 (No Loading)
```bash
./build_and_transfer_docker.sh -t voxtral-v1
```

### 2️⃣ Build & Load on Remote
```bash
./build_and_transfer_docker.sh -t voxtral-v1 --load
```

### 3️⃣ Build with GPU Support
```bash
./build_and_transfer_docker.sh -t gpu-v1 --gpu --load
```

### 4️⃣ Build & Push to Registry
```bash
./build_and_transfer_docker.sh -t latest --load -r docker.io/myuser
```

### 5️⃣ Using Profiles (Simpler)
```bash
./build_profiles.sh voxtral --load
./build_profiles.sh gpu --load -u myuser
./build_profiles.sh latest
```

## 📋 What Each Script Does

### Main Script: `build_and_transfer_docker.sh`

```bash
Step 1: Compile OVMS with C++ (uses Bazel)
        └─> Creates Docker image
        
Step 2: Compress image to .tar.gz
        └─> Saves to ./ovms-{tag}-{timestamp}.tar.gz
        
Step 3: Transfer via SCP to remote server
        └─> Copies to /tmp on remote
        
Step 4 (Optional): Load image with docker load
        └─> Extracts and registers with Docker
        
Step 5 (Optional): Tag & push to registry
        └─> docker push to Docker Hub or private registry
```

### Profiles: `build_profiles.sh`

Pre-configured combinations:

| Profile | Command | Result |
|---------|---------|--------|
| `quick` | `quick CPU build` | Fast iteration testing |
| `voxtral` | `CPU, Voxtral ready` | WebRTC + ASR support |
| `gpu` | `GPU-optimized` | All features on GPU |
| `latest` | `Full featured` | Production variant |
| `redhat` | `RedHat UBI9` | Enterprise Linux |
| `release` | `No cache, max jobs` | Clean build for release |

## ⚙️ Common Options

```
-t, --tag TAG              Set image version (default: latest)
-g, --gpu                  Build GPU support
-l, --load                 Load on remote after transfer
-r, --registry REG         Push to Docker Hub (e.g., docker.io/user)
-o, --os OS               Base OS: ubuntu24 (default), ubuntu22, redhat
-u, --user USER           SSH user for remote (default: current user)
-j, --jobs N              Build parallelism (default: auto)
--no-cache                Skip Docker cache
```

## 📍 File Locations

### Local (Your Machine)
```
~/model_server/build_and_transfer_docker.sh  ← Main script
~/model_server/build_profiles.sh             ← Convenience wrapper
~/model_server/BUILD_AND_TRANSFER_README.md  ← Full documentation
~/model_server/ovms-*.tar.gz                 ← Generated archives
```

### Remote (ov-arls-38)
```
/tmp/ovms-*.tar.gz         ← Default transfer location
(loaded into Docker)        ← After docker load
```

## 🔑 Prerequisites

**Local:**
- Docker installed and running
- Bash shell
- `make` and basic build tools
- SSH/SCP access to remote server

**Remote (ov-arls-38):**
- Docker installed and running  
- SSH access
- ~3-5 GB free disk space in `/tmp`
- (Can customize transfer dir with `-d` flag)

## ✨ Example Workflows

### Workflow 1: Test Voxtral Changes
```bash
# Build and load on remote
./build_and_transfer_docker.sh -t voxtral-test --load

# SSH to remote
ssh ov-arls-38

# Run with Voxtral
docker run -it \
  -e OVMS_WEBRTC_VOXTRAL_MODEL_PATH=/models/voxtral-realtime-2602 \
  -v /models:/models:ro \
  -p 19112:9112 \
  openvino/model_server:voxtral-test
```

### Workflow 2: Build for Production & Push to Registry
```bash
./build_and_transfer_docker.sh \
  -t 2026.5.0 \
  --load \
  -r docker.io/mycompany \
  -j 32 \
  --no-cache
```

### Workflow 3: Quick Iteration (No Remote Load)
```bash
# Just build and transfer (10-15 min)
./build_and_transfer_docker.sh -t dev-$(date +%s)

# Later, manually load on remote when ready:
ssh ov-arls-38 "cd /tmp && gunzip -c ovms-*.tar.gz | docker load"
```

## 🔍 Checking Progress

### During Build (Local)
```bash
# Watch Docker build output
docker images | grep openvino  # Check images
docker ps                       # Check running containers
```

### During Transfer
```bash
# In another terminal, check transfer progress
watch 'ls -lh ~/model_server/ovms-*.tar.gz'
```

### After Transfer (Remote)
```bash
ssh ov-arls-38

# Check if archive arrived
ls -lh /tmp/ovms-*.tar.gz

# Check if image was loaded
docker images | grep openvino

# Run the image
docker run -it openvino/model_server:voxtral-test /ovms/bin/ovms --help
```

## 🚨 Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| SSH connection fails | `ssh-keygen && ssh-copy-id ov-arls-38` |
| "No space left" | Check disk: `df -h` |
| Docker build hangs | Use fewer jobs: `-j 2` |
| Transfer slow | Check network: `ping ov-arls-38` |
| Image won't load | Ensure Docker running on remote: `ssh ov-arls-38 docker ps` |

## 📊 Typical Times

| Operation | Time | Notes |
|-----------|------|-------|
| Build (CPU) | 20-40 min | Depends on cache, hardware |
| Build (GPU) | 30-50 min | More dependencies |
| Archive size | 800MB-1.5GB | Compressed |
| Transfer (1Gbps) | 7-15 min | Depends on network |
| Load on remote | 2-5 min | Docker uncompression |
| **Total** | **35-70 min** | All steps |

## 🎯 Next Steps

1. **Read full docs:**
   ```bash
   cat BUILD_AND_TRANSFER_README.md
   ```

2. **Test the script:**
   ```bash
   ./build_and_transfer_docker.sh --help
   ```

3. **Run your first build:**
   ```bash
   ./build_and_transfer_docker.sh -t test --load
   ```

4. **Check on remote:**
   ```bash
   ssh ov-arls-38 docker images | grep openvino
   ```

## 📞 Support

For issues or questions:
1. Check `BUILD_AND_TRANSFER_README.md` troubleshooting section
2. Run with verbose: `bash -x ./build_and_transfer_docker.sh -t test`
3. Check Docker daemon: `docker ps`
4. Verify SSH: `ssh -v ov-arls-38`

---

**Happy building! 🐳**
