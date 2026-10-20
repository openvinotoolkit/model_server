#!/bin/bash

#############################################################################
# Build OVMS Docker Image with Local OpenVINO GenAI Branch
#############################################################################

set -e

set -a; 
source ~/.env; 
set +a;

export BAZEL_DEBUG_BUILD_FLAGS='" --output_filter=DONT_MATCH_ANYTHING_XYZ --define=CLOUD_DISABLE=1"'
# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show help
show_help() {
    cat << EOF
Usage: $0 [BASE_OS] [IMAGE_TAG]

Build OpenVINO Model Server with local OpenVINO GenAI branch.

ARGUMENTS:
    BASE_OS      Base OS for build (default: ubuntu24)
                 Options: ubuntu24, ubuntu22, redhat
    IMAGE_TAG    Docker image tag (default: krz-local-genai-<hash>-<subject>)

EXAMPLES:
    $0                              # Build ubuntu24 with default tag
    $0 ubuntu24                     # Explicit ubuntu24
    $0 redhat                       # Build RedHat variant
    $0 ubuntu22 my-custom-tag       # Ubuntu22 with custom tag

REQUIREMENTS:
    - Local openvino.genai repository at ../openvino.genai
    - Docker with BuildKit support

EOF
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Configuration with defaults
BASE_OS="${1:-ubuntu24}"
IMAGE_TAG="${2:-}"
if [[ -z "$IMAGE_TAG" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    COMMIT_HASH="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD)"
    COMMIT_SUBJECT="$(git -C "$SCRIPT_DIR" log -1 --format=%s | LC_ALL=C tr '[:upper:]' '[:lower:]' | LC_ALL=C sed -E 's/[^a-z0-9]+/-/g' | cut -c1-32 | sed -E 's/^-+//; s/-+$//')"
    IMAGE_TAG="krz-local-genai-${COMMIT_HASH}${COMMIT_SUBJECT:+-$COMMIT_SUBJECT}"
fi
GENAI_PATH="../openvino.genai"

# Verify OpenVINO GenAI repo exists
if [ ! -d "$GENAI_PATH" ]; then
    print_error "OpenVINO GenAI not found at: $GENAI_PATH"
    echo "Clone it with: git clone https://github.com/openvinotoolkit/openvino.genai $GENAI_PATH"
    exit 1
fi

print_info "=========================================="
print_info "Build OVMS with Local OpenVINO GenAI"
print_info "=========================================="
print_info "Base OS: $BASE_OS"
print_info "Image Tag: $IMAGE_TAG"
print_info "GenAI Path: $GENAI_PATH"
echo ""

# Change to model_server directory
cd "$(dirname "${BASH_SOURCE[0]}")"

print_info "Step 1: Copying OpenVINO GenAI to build context..."
if [ -d "./openvino_genai_src" ]; then
    rm -rf ./openvino_genai_src
fi
cp -r "$GENAI_PATH" ./openvino_genai_src
print_success "OpenVINO GenAI copied"

print_info "Step 2: Building Docker image..."
print_info "Executing: DOCKER_BUILDKIT=1 make docker_build BASE_OS=$BASE_OS OVMS_CPP_IMAGE_TAG=$IMAGE_TAG OV_USE_BINARY=0"
print_info "Suppressing deprecation warnings to allow GenAI source build..."

# Build using Makefile (handles BASE_IMAGE and all other args)
# Enable BuildKit for mount support
# Suppress deprecation warnings which are breaking the GenAI build from source

echo $http_proxy, $https_proxy

BUILD_LOG="/tmp/build_with_local_genai_$(date +%s).log"
print_info "Full build output will be logged to: $BUILD_LOG"

# BuildKit truncates each RUN step's log at ~1MiB; bazel compiler warnings alone can exceed that
# and hide the real error. Suppress them via bazel's --output_filter (errors still show).
# Must be exported (not passed as a make CLI var) so the Makefile's "?=" default doesn't strip
# the leading space and mangle the flag.
# NOTE: avoid a value ending in "$" right before a quote - bash reads `$"` as its
# gettext-translated-string syntax and mangles the surrounding quoting.
# NOTE: the Makefile glues BAZEL_DEBUG_BUILD_FLAGS between two other literally-quoted
# strings (e.g. "--strip="always"" + BAZEL_DEBUG_BUILD_FLAGS + " --config=..."), relying on
# embedded `"` characters (not real shell quoting) to keep it all one shell word. A plain
# string with a leading space breaks that and leaks a stray word to the docker CLI, so this
# value must itself start/end with a literal `"` character like the other *_PARAMS vars do.
# CLOUD_DISABLE=1 skips fetching aws-sdk-cpp/azure/google-cloud-cpp (no S3/GCS/Azure model paths).
export BAZEL_DEBUG_BUILD_FLAGS='" --output_filter=DONT_MATCH_ANYTHING_XYZ --define=CLOUD_DISABLE=1"'
export MEDIAPIPE_DISABLE=0


DOCKER_BUILDKIT=1 make docker_build \
    BASE_OS=$BASE_OS \
    MEDIAPIPE_DISABLE=0 \
    OVMS_CPP_IMAGE_TAG=$IMAGE_TAG \
    OV_USE_BINARY=0 \
    TOKEN="$GITHUB_TOKEN" 2>&1 | tee "$BUILD_LOG"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Docker image built successfully"
    FULL_IMAGE="openvino/model_server:$IMAGE_TAG"
    echo ""
    print_info "Image: $FULL_IMAGE"
    print_info "You can now transfer it with:"
    print_info "  ./build_and_transfer_docker.sh -t $IMAGE_TAG --load"
    print_info "Full build log: $BUILD_LOG"
else
    print_error "Build failed!"
    print_info "Full build log: $BUILD_LOG"
    exit 1
fi

print_info "Step 3: Cleaning up build context..."
rm -rf ./openvino_genai_src
print_success "Cleanup complete"

echo ""
print_success "Build complete!"
