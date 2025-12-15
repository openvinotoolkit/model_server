#!/bin/bash
set -e  # Exit immediately if any command fails
set -u  # Exit if undefined variables are used
set -o pipefail  # Exit if any command in a pipeline fails

# Save original directory
ORIGINAL_DIR=$(pwd)

# Function to handle errors and cleanup
error_handler() {
    echo "Error: Command failed at line $1" >&2
    cd "$ORIGINAL_DIR"
    exit 1
}

# Function to cleanup on exit
cleanup() {
    cd "$ORIGINAL_DIR"
}

# Trap errors and call error_handler
trap 'error_handler $LINENO' ERR
trap cleanup EXIT

# Change to /openvino_genai directory
cd /openvino_genai

echo "Starting OpenVINO GenAI build..."

# Export compiler flags
export SDL_OPS="-fpic -O2 -U_FORTIFY_SOURCE -fstack-protector -fno-omit-frame-pointer -D_FORTIFY_SOURCE=1 -fno-strict-overflow -Wall -Wno-unknown-pragmas -Wno-error=sign-compare -fno-delete-null-pointer-checks -fwrapv -fstack-clash-protection -Wformat -Wformat-security -Werror=format-security"

# Run CMake configuration
echo "Configuring CMake..."
if ! cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${SDL_OPS}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DENABLE_SYSTEM_ICU="True" \
    -DBUILD_TOKENIZERS=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_TOOLS=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_XGRAMMAR=ON \
    -S ./ \
    -B ./build/; then
    echo "Error: CMake configuration failed" >&2
    exit 1
fi

# Build the project
echo "Building project..."
if ! cmake --build ./build/ --parallel ${JOBS:-$(nproc)}; then
    echo "Error: CMake build failed" >&2
    exit 1
fi

# Copy shared libraries
echo "Copying shared libraries..."
if ! cp /openvino_genai/build/openvino_genai/lib*.so* /opt/intel/openvino/runtime/lib/intel64/; then
    echo "Error: Failed to copy shared libraries" >&2
    exit 1
fi

# Copy header files
echo "Copying header files..."
if ! cp -r /openvino_genai/src/cpp/include/* /opt/intel/openvino/runtime/include/; then
    echo "Error: Failed to copy header files" >&2
    exit 1
fi

# Copy Python files
echo "Copying Python files..."
if ! cp -r /openvino_genai/build/openvino_genai/*py* /opt/intel/openvino/python/; then
    echo "Error: Failed to copy Python files" >&2
    exit 1
fi

echo 'Built successfully'