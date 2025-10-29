# Summary of Changes for TensorFlow Serving Proto Dependency

## Overview
This change replaces the dependency on the entire TensorFlow Serving and TensorFlow repositories with a dependency on just the protobuf files required to create a TensorFlow Serving compatible service.

## Changes Made

### 1. Created `third_party/tensorflow_serving_protos/` Directory
- **Purpose**: Contains only the `.proto` files needed for TensorFlow Serving API compatibility
- **Contents**:
  - `tensorflow_serving/apis/` - TensorFlow Serving API proto files (15 files)
  - `tensorflow_serving/config/` - TensorFlow Serving config proto files (7 files)
  - `tensorflow/core/framework/` - TensorFlow framework proto files (30 files)
  - `tensorflow/core/example/` - TensorFlow example proto files (3 files)
  - `tensorflow/core/protobuf/` - TensorFlow protobuf definitions (41 files)
- **Total**: 96 proto files extracted from TensorFlow Serving 2.18.0 and TensorFlow commit 5329ec8d

### 2. Created `third_party/tensorflow_serving_protos/BUILD`
- **Purpose**: Bazel build rules for compiling the proto files
- **Targets**:
  - `tensorflow_core_framework_cc_proto` - TensorFlow framework protos
  - `tensorflow_core_example_cc_proto` - TensorFlow example protos
  - `tensorflow_core_protobuf_cc_proto` - TensorFlow protobuf protos
  - `tensorflow_serving_config_cc_proto` - TF Serving config protos
  - `tensorflow_serving_apis_cc_proto` - TF Serving API protos
  - `prediction_service_cc_proto` - Prediction service with gRPC
  - `model_service_cc_proto` - Model service with gRPC
- **Style**: Uses `@com_google_protobuf//:protobuf.bzl` consistent with existing codebase

### 3. Updated `WORKSPACE`
- Added `new_local_repository` for `tensorflow_serving_protos`
- Placed before the `tensorflow_serving` git repository definition
- Uses BUILD file from `third_party/tensorflow_serving_protos/BUILD`

### 4. Updated `BUILD.bazel`
- Changed proto dependencies in `ovms_dependencies`:
  - FROM: `@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto`
  - TO: `@tensorflow_serving_protos//:prediction_service_cc_proto`
  - FROM: `@tensorflow_serving//tensorflow_serving/apis:model_service_cc_proto`
  - TO: `@tensorflow_serving_protos//:model_service_cc_proto`
- **Kept**: Utility dependencies on `tensorflow_serving/util:json_tensor` (still needed for JSON conversion)

### 5. Updated `src/BUILD`
- Updated `tfs_utils` target:
  - Changed from individual API proto deps to single `tensorflow_serving_apis_cc_proto`
- Updated `libovmshttpservermodule` target:
  - Changed from `prediction_service_cc_proto` to `tensorflow_serving_apis_cc_proto`
- **Kept**: Utility dependencies on `threadpool_executor`, `json_tensor`, `net_http` (still needed)

### 6. Documentation
- Created `third_party/tensorflow_serving_protos/README.md` explaining:
  - Contents of the directory
  - Source versions (TF Serving 2.18.0, TF commit 5329ec8d)
  - Usage instructions
  - Rationale for the approach
  - Maintenance guide
- Created `third_party/tensorflow_serving_protos/.gitignore` to exclude generated files

## Benefits

1. **Reduced Build Dependencies**: No longer need to build entire TensorFlow or TensorFlow Serving
2. **Faster Builds**: Only proto compilation needed, not C++ library compilation
3. **Easier Maintenance**: Proto files are simpler to update than full framework dependencies
4. **Bazel-native**: All proto compilation happens in Bazel, not in Dockerfiles
5. **API Compatibility**: Maintains full TensorFlow Serving API compatibility

## What Was NOT Changed

1. **Utility Dependencies**: Still depend on TensorFlow Serving for:
   - `tensorflow_serving/util:json_tensor` - JSON tensor conversion utilities
   - `tensorflow_serving/util:threadpool_executor` - Thread pool executor
   - `tensorflow_serving/util/net_http/server/public:http_server` - HTTP server
   
   These are C++ utilities, not protos, and are still needed for functionality.

2. **TensorFlow Core Framework**: Still depend on `@org_tensorflow//tensorflow/core:framework` for Eigen Tensor support

## Testing Required

1. Build verification: Ensure the project builds successfully with new dependencies
2. Functional testing: Verify TensorFlow Serving API endpoints work correctly
3. Integration testing: Test with existing TF Serving clients
4. Performance testing: Ensure no performance regression

## Future Work

1. Consider extracting utility files (`json_tensor`, `threadpool_executor`, `net_http`) to reduce dependency on full TensorFlow Serving repository
2. Monitor TensorFlow Serving releases for proto file changes
3. Create automated script to update proto files when versions change
