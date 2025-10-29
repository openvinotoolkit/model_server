# TensorFlow Serving Protos

This directory contains protocol buffer (`.proto`) files extracted from TensorFlow and TensorFlow Serving repositories. These proto files are used to provide TensorFlow Serving API compatibility without requiring the full TensorFlow or TensorFlow Serving build dependencies.

## Contents

### TensorFlow Serving APIs (`tensorflow_serving/apis/`)
Proto files defining the TensorFlow Serving gRPC API surface:
- `prediction_service.proto` - Main prediction service API
- `model_service.proto` - Model management service API
- `predict.proto`, `classify.proto`, `regress.proto` - Request/response messages
- `model.proto`, `input.proto` - Common types
- Other supporting proto files

### TensorFlow Serving Config (`tensorflow_serving/config/`)
Configuration proto files for TensorFlow Serving:
- `model_server_config.proto` - Model server configuration
- `logging_config.proto` - Logging configuration
- Other config protos

### TensorFlow Core (`tensorflow/core/`)
Required TensorFlow proto dependencies:
- `framework/*.proto` - Core TensorFlow framework types (tensor, types, etc.)
- `example/*.proto` - TensorFlow Example proto format
- `protobuf/*.proto` - TensorFlow protobuf definitions

## Source Information

- **TensorFlow Serving Version**: 2.18.0
- **TensorFlow Version**: Commit 5329ec8dd396487982ef3e743f98c0195af39a6b

## Usage

The BUILD file in this directory provides Bazel targets for compiling these protos:

```python
# For TensorFlow Serving APIs (non-service protos)
deps = ["@tensorflow_serving_protos//:tensorflow_serving_apis_cc_proto"]

# For Prediction Service (with gRPC)
deps = ["@tensorflow_serving_protos//:prediction_service_cc_proto"]

# For Model Service (with gRPC)
deps = ["@tensorflow_serving_protos//:model_service_cc_proto"]
```

## Rationale

This approach allows OpenVINO Model Server to:
1. Maintain TensorFlow Serving API compatibility
2. Avoid building the entire TensorFlow and TensorFlow Serving stack
3. Reduce build complexity and time
4. Keep proto compilation within Bazel (not in Dockerfiles)

## Maintenance

When updating to a new TensorFlow Serving version:
1. Clone the appropriate TensorFlow Serving tag
2. Clone the matching TensorFlow commit
3. Copy the required proto files from both repositories
4. Update this README with version information
5. Test the build to ensure compatibility
