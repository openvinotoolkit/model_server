//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "status.hpp"

namespace ovms {

const std::unordered_map<const StatusCode, const std::string> Status::statusMessageMap = {
    {StatusCode::OK, ""},

    {StatusCode::PATH_INVALID, "The provided base path is invalid or doesn't exists"},
    {StatusCode::FILE_INVALID, "File not found or cannot open"},
    {StatusCode::CONFIG_FILE_INVALID, "Configuration file not found or cannot open"},
    {StatusCode::FILESYSTEM_ERROR, "Error during filesystem operation"},
    {StatusCode::NOT_IMPLEMENTED, "Functionality not implemented"},
    {StatusCode::NO_MODEL_VERSION_AVAILABLE, "Not a single model version directory has valid numeric name"},
    {StatusCode::MODEL_NOT_LOADED, "Error while loading a model"},
    {StatusCode::JSON_INVALID, "The file is not valid json"},
    {StatusCode::JSON_SERIALIZATION_ERROR, "Data serialization to json format failed"},
    {StatusCode::MODELINSTANCE_NOT_FOUND, "ModelInstance not found"},
    {StatusCode::SHAPE_WRONG_FORMAT, "The provided shape is in wrong format"},
    {StatusCode::LAYOUT_WRONG_FORMAT, "The provided layout is in wrong format"},
    {StatusCode::DIM_WRONG_FORMAT, "The provided dimension is in wrong format"},
    {StatusCode::PLUGIN_CONFIG_WRONG_FORMAT, "Plugin config is in wrong format"},
    {StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT, "Model version policy is in wrong format"},
    {StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY, "Model version policy contains unsupported key"},
    {StatusCode::GRPC_CHANNEL_ARG_WRONG_FORMAT, "Grpc channel arguments passed in wrong format"},
    {StatusCode::CONFIG_FILE_TIMESTAMP_READING_FAILED, "Error during config file timestamp reading"},
    {StatusCode::RESHAPE_ERROR, "Model could not be reshaped with requested shape"},
    {StatusCode::RESHAPE_REQUIRED, "Model needs to be reloaded with new shape"},
    {StatusCode::BATCHSIZE_CHANGE_REQUIRED, "Model needs to be reloaded with new batchsize"},
    {StatusCode::FORBIDDEN_MODEL_DYNAMIC_PARAMETER, "Value of provided parameter is forbidden"},
    {StatusCode::ANONYMOUS_FIXED_SHAPE_NOT_ALLOWED, "Anonymous fixed shape is invalid for models with multiple inputs"},
    {StatusCode::ANONYMOUS_FIXED_LAYOUT_NOT_ALLOWED, "Anonymous fixed layout is invalid for models with multiple inputs"},
    {StatusCode::CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE, "Cannot compile model into target device"},
    {StatusCode::MODEL_MISSING, "Model with requested name and/or version is not found"},
    {StatusCode::MODEL_CONFIG_INVALID, "Model config is invalid"},
    {StatusCode::MODEL_NAME_MISSING, "Model with requested name is not found"},
    {StatusCode::MODEL_NAME_OCCUPIED, "Given model name is already occupied"},
    {StatusCode::MODEL_VERSION_MISSING, "Model with requested version is not found"},
    {StatusCode::MODEL_VERSION_NOT_LOADED_ANYMORE, "Model with requested version is retired"},
    {StatusCode::MODEL_VERSION_NOT_LOADED_YET, "Model with requested version is not loaded yet"},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_ANYMORE, "Pipeline is retired"},
    {StatusCode::PIPELINE_DEFINITION_NOT_LOADED_YET, "Pipeline is not loaded yet"},
    {StatusCode::MODEL_SPEC_MISSING, "model_spec missing in request"},
    {StatusCode::MODEL_VERSION_INVALID_FORMAT, "invalid model version format in request"},
    {StatusCode::INVALID_SIGNATURE_DEF, "Invalid signature name"},
    {StatusCode::CONFIG_SHAPE_IS_NOT_IN_MODEL, "Shape from config not found in model"},
    {StatusCode::CONFIG_LAYOUT_IS_NOT_IN_MODEL, "Layout from config not found in model"},
    {StatusCode::CONFIG_SHAPE_MAPPED_BUT_USED_REAL_NAME, "Shape from config has real name. Use mapped name instead"},
    {StatusCode::CONFIG_LAYOUT_MAPPED_BUT_USED_REAL_NAME, "Layout from config has real name. Use mapped name instead"},
    {StatusCode::INVALID_NIREQ, "Nireq parameter too high"},
    {StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_SUBSCRIBED_MODEL, "Requested dynamic parameters but model is used in pipeline"},
    {StatusCode::PIPELINE_STREAM_ID_NOT_READY_YET, "Node is not ready for execution"},
    {StatusCode::REQUESTED_DYNAMIC_PARAMETERS_ON_STATEFUL_MODEL, "Dynamic shape and dynamic batch size are not supported for stateful models"},
    {StatusCode::REQUESTED_STATEFUL_PARAMETERS_ON_SUBSCRIBED_MODEL, "Stateful model cannot be subscribed to pipeline"},
    {StatusCode::REQUESTED_MODEL_TYPE_CHANGE, "Model type cannot be changed after it is loaded"},
    {StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER, "Stateful model config parameter used for non stateful model"},
    {StatusCode::INVALID_MAX_SEQUENCE_NUMBER, "Sequence max number parameter too high"},
    {StatusCode::CANNOT_CONVERT_FLAT_SHAPE, "Cannot convert flat shape to Shape object"},
    {StatusCode::INVALID_BATCH_DIMENSION, "Invalid batch dimension in shape"},
    {StatusCode::LAYOUT_INCOMPATIBLE_WITH_SHAPE, "Layout incompatible with given shape"},
    {StatusCode::ALLOW_CACHE_WITH_CUSTOM_LOADER, "allow_cache is set to true with custom loader usage"},
    {StatusCode::UNKNOWN_ERROR, "Unknown error"},

    // Sequence management
    {StatusCode::SEQUENCE_MISSING, "Sequence with provided ID does not exist"},
    {StatusCode::SEQUENCE_ALREADY_EXISTS, "Sequence with provided ID already exists"},
    {StatusCode::SEQUENCE_ID_NOT_PROVIDED, "Sequence ID has not been provided in request inputs"},
    {StatusCode::INVALID_SEQUENCE_CONTROL_INPUT, "Unexpected value of sequence control input"},
    {StatusCode::SEQUENCE_ID_BAD_TYPE, "Could not find sequence id in expected tensor proto field uint64_val"},
    {StatusCode::SEQUENCE_CONTROL_INPUT_BAD_TYPE, "Could not find sequence control input in expected tensor proto field uint32_val"},
    {StatusCode::SEQUENCE_TERMINATED, "Sequence last request is being processed and it's not available anymore"},
    {StatusCode::SPECIAL_INPUT_NO_TENSOR_SHAPE, "Special input proto does not contain tensor shape information"},
    {StatusCode::MAX_SEQUENCE_NUMBER_REACHED, "Max sequence number has been reached. Could not create new sequence."},

    // Predict request validation
    {StatusCode::INVALID_NO_OF_INPUTS, "Invalid number of inputs"},
    {StatusCode::INVALID_MISSING_INPUT, "Missing input with specific name"},
    {StatusCode::INVALID_MISSING_OUTPUT, "Missing output with specific name"},
    {StatusCode::INVALID_NO_OF_SHAPE_DIMENSIONS, "Invalid number of shape dimensions"},
    {StatusCode::INVALID_BATCH_SIZE, "Invalid input batch size"},
    {StatusCode::INVALID_SHAPE, "Invalid input shape"},
    {StatusCode::INVALID_BUFFER_TYPE, "Invalid input buffer type"},
    {StatusCode::INVALID_DEVICE_ID, "Invalid input buffer device id"},
    {StatusCode::INVALID_STRING_INPUT, "Invalid string input"},
    {StatusCode::INVALID_STRING_MAX_SIZE_EXCEEDED, "Maximum 2D array after string conversion exceeded 1GB"},
    {StatusCode::INVALID_INPUT_FORMAT, "Inputs inside buffer does not match expected format."},
    {StatusCode::INVALID_PRECISION, "Invalid input precision"},
    {StatusCode::INVALID_VALUE_COUNT, "Invalid number of values in tensor proto container"},
    {StatusCode::INVALID_CONTENT_SIZE, "Invalid content size of tensor proto"},
    {StatusCode::INVALID_MESSAGE_STRUCTURE, "Passing buffers both in ModelInferRequest::InferInputTensor::contents and in ModelInferRequest::raw_input_contents is not allowed"},
    {StatusCode::UNSUPPORTED_LAYOUT, "Received binary image input but resource not configured to accept NHWC layout"},

    // Deserialization
    {StatusCode::OV_UNSUPPORTED_DESERIALIZATION_PRECISION, "Unsupported deserialization precision"},
    {StatusCode::OV_INTERNAL_DESERIALIZATION_ERROR, "Internal deserialization error"},

    // Inference
    {StatusCode::OV_INTERNAL_INFERENCE_ERROR, "Internal inference error"},

    // Serialization
    {StatusCode::OV_UNSUPPORTED_SERIALIZATION_PRECISION, "Unsupported serialization precision"},
    {StatusCode::OV_INTERNAL_SERIALIZATION_ERROR, "Internal serialization error"},
    {StatusCode::OV_CLONE_TENSOR_ERROR, "Error during tensor clone"},

    // GetModelStatus
    {StatusCode::INTERNAL_ERROR, "Internal server error"},

    // Rest handler failure
    {StatusCode::REST_NOT_FOUND, "Requested REST resource not found"},
    {StatusCode::REST_COULD_NOT_PARSE_VERSION, "Could not parse model version in request"},
    {StatusCode::REST_INVALID_URL, "Invalid request URL"},
    {StatusCode::REST_UNSUPPORTED_METHOD, "Unsupported method"},
    {StatusCode::UNKNOWN_REQUEST_COMPONENTS_TYPE, "Request components type not recognized"},

    // Rest parser failure
    {StatusCode::REST_BODY_IS_NOT_AN_OBJECT, "Request body should be JSON object"},
    {StatusCode::REST_PREDICT_UNKNOWN_ORDER, "Invalid JSON structure. Could not detect row or column format"},
    {StatusCode::REST_INSTANCES_NOT_AN_ARRAY, "Invalid JSON structure. Nonamed instance is not an array."},
    {StatusCode::REST_NAMED_INSTANCE_NOT_AN_OBJECT, "Invalid JSON structure. One of named instances is not a JSON object."},
    {StatusCode::REST_INPUT_NOT_PREALLOCATED, "Internal allocation error"},
    {StatusCode::REST_NO_INSTANCES_FOUND, "Invalid JSON structure. Missing instances in row format"},
    {StatusCode::REST_INSTANCES_NOT_NAMED_OR_NONAMED, "Could not detect neither named or nonamed format"},
    {StatusCode::REST_COULD_NOT_PARSE_INSTANCE, "Could not parse instance content. Not valid ndarray detected"},
    {StatusCode::REST_INSTANCES_BATCH_SIZE_DIFFER, "Invalid JSON structure. Request inputs have different batch sizes"},
    {StatusCode::REST_INPUTS_NOT_AN_OBJECT, "Invalid JSON structure. One of inputs is not a JSON object."},
    {StatusCode::REST_NO_INPUTS_FOUND, "Invalid JSON structure, missing inputs"},
    {StatusCode::REST_COULD_NOT_PARSE_INPUT, "Could not parse input content. Not valid ndarray detected"},
    {StatusCode::REST_COULD_NOT_PARSE_OUTPUT, "Could not parse output content."},
    {StatusCode::REST_COULD_NOT_PARSE_PARAMETERS, "Could not parse request parameters"},
    {StatusCode::REST_PROTO_TO_STRING_ERROR, "Response parsing to JSON error"},
    {StatusCode::REST_BASE64_DECODE_ERROR, "Decode Base64 to string error"},
    {StatusCode::REST_UNSUPPORTED_PRECISION, "Could not parse input content. Unsupported data precision detected"},
    {StatusCode::REST_SERIALIZE_TENSOR_CONTENT_INVALID_SIZE, "Size of data in tensor_content does not match declared tensor shape"},
    {StatusCode::REST_SERIALIZE_VAL_FIELD_INVALID_SIZE, "Number of elements in xxx_val field does not match declared tensor shape"},
    {StatusCode::REST_SERIALIZE_NO_DATA, "No data found in tensor_content or xxx_val field matching tensor dtype"},
    {StatusCode::REST_BINARY_DATA_SIZE_PARAMETER_INVALID, "binary_data_size parameter is invalid and cannot be parsed"},
    {StatusCode::REST_BINARY_BUFFER_EXCEEDED, "Received buffer size is smaller than binary_data_size parameter indicates"},
    {StatusCode::REST_INFERENCE_HEADER_CONTENT_LENGTH_INVALID, "Inference-Header-Content-Length header is invalid and couldn't be parsed"},
    {StatusCode::REST_CONTENTS_FIELD_NOT_EMPTY, "Request contains values both in binary data and in content value"},

    // Pipeline validation errors
    {StatusCode::PIPELINE_DEFINITION_ALREADY_EXIST, "Pipeline definition with the same name already exists"},
    {StatusCode::PIPELINE_NODE_WRONG_KIND_CONFIGURATION, "Unsupported node type"},
    {StatusCode::PIPELINE_MULTIPLE_ENTRY_NODES, "Pipeline definition has multiple request nodes"},
    {StatusCode::PIPELINE_MULTIPLE_EXIT_NODES, "Pipeline definition has multiple response nodes"},
    {StatusCode::PIPELINE_MISSING_ENTRY_OR_EXIT, "Pipeline definition is missing request or response node"},
    {StatusCode::PIPELINE_DEFINITION_NAME_MISSING, "Model with requested name is not found"},
    {StatusCode::PIPELINE_NODE_NAME_DUPLICATE, "Pipeline definition has multiple nodes with the same name"},
    {StatusCode::PIPELINE_CYCLE_FOUND, "Pipeline definition contains a cycle"},
    {StatusCode::PIPELINE_CONTAINS_UNCONNECTED_NODES, "Pipeline definition has unconnected nodes"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_NODE, "Pipeline definition has reference to missing node"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL, "Pipeline definition has reference to missing model"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_DATA_SOURCE, "Pipeline definition has reference to missing data source"},
    {StatusCode::PIPELINE_NODE_REFERING_TO_MISSING_MODEL_OUTPUT, "Pipeline definition has reference to missing model output"},
    {StatusCode::PIPELINE_CONNECTION_TO_MISSING_MODEL_INPUT, "Pipeline definition has connection to non existing model input"},
    {StatusCode::PIPELINE_NOT_ALL_INPUTS_CONNECTED, "Pipeline definition does not have connections for all inputs of underlying models"},
    {StatusCode::PIPELINE_MODEL_INPUT_CONNECTED_TO_MULTIPLE_DATA_SOURCES, "Pipeline definition has multiple connections to the same input of underlying model"},
    {StatusCode::PIPELINE_EXIT_USED_AS_NODE_DEPENDENCY, "Pipeline definition has response node used as dependency node"},
    {StatusCode::PIPELINE_NAME_OCCUPIED, "Pipeline has the same name as model"},
    {StatusCode::PIPELINE_DEFINITION_INVALID_NODE_LIBRARY, "Pipeline refers to incorrect library"},
    {StatusCode::PIPELINE_INCONSISTENT_SHARD_DIMENSIONS, "Gathered tensor shards dimensions are different"},
    {StatusCode::PIPELINE_WRONG_NUMBER_OF_DIMENSIONS_TO_DEMULTIPLY, "Wrong number of dimensions in a tensor to be sharded"},
    {StatusCode::PIPELINE_WRONG_DIMENSION_SIZE_TO_DEMULTIPLY, "Wrong dimension size. Should match demultiply count"},
    {StatusCode::PIPELINE_TRIED_TO_SET_THE_SAME_INPUT_TWICE, "Tried to set the same input twice for node input handler"},
    {StatusCode::PIPELINE_TRIED_TO_SET_INPUT_SHARD_FOR_ORDINARY_INPUT_HANDLER, "Tried to set input with shard id > 0 for ordinary input handler"},
    {StatusCode::PIPELINE_NODE_GATHER_FROM_NOT_EXISTING_NODE, "Gather node refers to not existing node"},
    {StatusCode::PIPELINE_NODE_GATHER_FROM_NOT_DEMULTIPLEXER, "Gather node refers to node that isn't demultiplexer"},
    {StatusCode::PIPELINE_NODE_GATHER_FROM_ENTRY_NODE, "Gathering from entry node is not allowed"},
    {StatusCode::PIPELINE_DEMULTIPLY_ENTRY_NODE, "Demultiplication at entry node is not allowed"},
    {StatusCode::PIPELINE_DEMULTIPLY_COUNT_DOES_NOT_MATCH_TENSOR_SHARD_COUNT, "Demultiplication count does not match tensor first dimension"},
    {StatusCode::PIPELINE_MANUAL_GATHERING_FROM_MULTIPLE_NODES_NOT_SUPPORTED, "Manual gathering from multiple nodes is not supported"},
    {StatusCode::PIPELINE_NOT_ENOUGH_SHAPE_DIMENSIONS_TO_DEMULTIPLY, "Pipeline has not enough shape dimensions to demultiply"},
    {StatusCode::PIPELINE_TOO_LARGE_DIMENSION_SIZE_TO_DEMULTIPLY, "Too large dynamic demultiplication requested."},
    {StatusCode::PIPELINE_WRONG_DEMULTIPLEXER_GATHER_NODES_ORDER, "Demultiplexer and gather nodes are not in LIFO order"},
    {StatusCode::PIPELINE_DEMULTIPLEXER_NO_RESULTS, "Pipeline execution aborted due to no content from custom node"},
    {StatusCode::PIPELINE_INPUTS_AMBIGUOUS_METADATA, "Multiple nodes connected to the same pipeline input require different tensor metadata"},

    // Mediapipe
    {StatusCode::MEDIAPIPE_GRAPH_START_ERROR, "Failed to start mediapipe graph"},
    {StatusCode::MEDIAPIPE_GRAPH_CONFIG_FILE_INVALID, "Failed to read protobuf graph configuration file"},
    {StatusCode::MEDIAPIPE_GRAPH_INITIALIZATION_ERROR, "Failed to initalize mediapipe graph"},
    {StatusCode::MEDIAPIPE_GRAPH_ADD_OUTPUT_STREAM_ERROR, "Failed to add mediapipe graph output stream"},
    {StatusCode::MEDIAPIPE_GRAPH_CLOSE_INPUT_STREAM_ERROR, "Failed to close mediapipe graph input stream"},
    {StatusCode::MEDIAPIPE_GRAPH_ADD_PACKET_INPUT_STREAM, "Failed to add packet to mediapipe graph input stream"},  // TODO add http/gRPC conversions for retCodes of Mediapipe
    {StatusCode::MEDIAPIPE_DEFINITION_NAME_MISSING, "Model with requested name is not found"},
    {StatusCode::MEDIAPIPE_EXECUTION_ERROR, "Mediapipe execution failed. MP status"},
    {StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_ANYMORE, "Mediapipe is retired"},
    {StatusCode::MEDIAPIPE_DEFINITION_NOT_LOADED_YET, "Mediapipe is not loaded yet"},

    // Storage errors
    // S3
    {StatusCode::S3_BUCKET_NOT_FOUND, "S3 Bucket not found"},
    {StatusCode::S3_METADATA_FAIL, "S3 metadata failure"},
    {StatusCode::S3_FAILED_LIST_OBJECTS, "S3 Failed to list objects"},
    {StatusCode::S3_FAILED_GET_TIME, "S3 Failed to get modification time"},
    {StatusCode::S3_INVALID_ACCESS, "S3 Invalid access rights"},
    {StatusCode::S3_FILE_NOT_FOUND, "S3 File or directory not found"},
    {StatusCode::S3_FILE_INVALID, "S3 File path is invalid"},
    {StatusCode::S3_FAILED_GET_OBJECT, "S3 Failed to get object from path"},

    // GCS
    {StatusCode::GCS_BUCKET_NOT_FOUND, "GCS Bucket not found"},
    {StatusCode::GCS_METADATA_FAIL, "GCS metadata failure"},
    {StatusCode::GCS_FAILED_LIST_OBJECTS, "GCS Failed to list objects"},
    {StatusCode::GCS_FAILED_GET_TIME, "GCS Failed to list objects"},
    {StatusCode::GCS_INVALID_ACCESS, "GCS Invalid access rights"},
    {StatusCode::GCS_FILE_NOT_FOUND, "GCS File or directory not found"},
    {StatusCode::GCS_FILE_INVALID, "GCS File path is invalid"},
    {StatusCode::GCS_FAILED_GET_OBJECT, "GCS Failed to get object from path"},
    {StatusCode::GCS_INCORRECT_REQUESTED_OBJECT_TYPE, "GCS invalid object type in path"},

    // AS
    {StatusCode::AS_INVALID_PATH, "AS Invalid path"},
    {StatusCode::AS_CONTAINER_NOT_FOUND, "AS Container not found"},
    {StatusCode::AS_SHARE_NOT_FOUND, "AS Share not found"},
    {StatusCode::AS_METADATA_FAIL, "AS metadata failure"},
    {StatusCode::AS_FAILED_LIST_OBJECTS, "AS Failed to list objects"},
    {StatusCode::AS_FAILED_GET_TIME, "AS Failed to list objects"},
    {StatusCode::AS_INVALID_ACCESS, "AS Invalid access rights"},
    {StatusCode::AS_FILE_NOT_FOUND, "AS File or directory not found"},
    {StatusCode::AS_FILE_INVALID, "AS File path is invalid"},
    {StatusCode::AS_FAILED_GET_OBJECT, "AS Failed to get object from path"},
    {StatusCode::AS_INCORRECT_REQUESTED_OBJECT_TYPE, "AS invalid object type in path"},

    // Custom Loader
    {StatusCode::CUSTOM_LOADER_LIBRARY_INVALID, "Custom Loader library not found or cannot open"},
    {StatusCode::CUSTOM_LOADER_LIBRARY_LOAD_FAILED, "Cannot load the custom library"},
    {StatusCode::CUSTOM_LOADER_EXISTS, "The custom loader is already present in loaders list"},
    {StatusCode::CUSTOM_LOADER_NOT_PRESENT, "The custom loader is not present in loaders list"},
    {StatusCode::CUSTOM_LOADER_INIT_FAILED, "Custom Loader LoadInit failed"},
    {StatusCode::CUSTOM_LOADER_ERROR, "Custom Loader Generic / Unknown Error"},

    // Custom Node
    {StatusCode::NODE_LIBRARY_ALREADY_LOADED, "Custom node library is already loaded"},
    {StatusCode::NODE_LIBRARY_LOAD_FAILED_OPEN, "Custom node library failed to open"},
    {StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM, "Custom node library failed to load symbol"},
    {StatusCode::NODE_LIBRARY_MISSING, "Custom node library not found"},
    {StatusCode::NODE_LIBRARY_MISSING_OUTPUT, "Custom node output is missing"},
    {StatusCode::NODE_LIBRARY_EXECUTION_FAILED, "Custom node failed during execution"},
    {StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED, "Custom node library has returned corrupted outputs handle"},
    {StatusCode::NODE_LIBRARY_OUTPUTS_CORRUPTED_COUNT, "Custom node library has produced corrupted number of outputs"},
    {StatusCode::NODE_LIBRARY_INVALID_PRECISION, "Custom node has produced tensor with unspecified precision"},
    {StatusCode::NODE_LIBRARY_INVALID_SHAPE, "Custom node has produced tensor with not matching shape"},
    {StatusCode::NODE_LIBRARY_INVALID_CONTENT_SIZE, "Custom node output has invalid content size"},
    {StatusCode::NODE_LIBRARY_METADATA_FAILED, "Custom node failed on metadata call"},
    {StatusCode::NODE_LIBRARY_OUTPUT_MISSING_NAME, "Custom node output is missing name"},

    // Binary inputs
    {StatusCode::IMAGE_PARSING_FAILED, "Image parsing failed"},
    {StatusCode::INVALID_NO_OF_CHANNELS, "Invalid number of channels in binary input"},
    {StatusCode::BINARY_IMAGES_RESOLUTION_MISMATCH, "Binary input images for this endpoint are required to have the same resolution"},
    {StatusCode::STRING_VAL_EMPTY, "String val is empty"},
    {StatusCode::BYTES_CONTENTS_EMPTY, "Bytes contents is empty"},
    {StatusCode::NODE_LIBRARY_INITIALIZE_FAILED, "Failure during custom node library initialization"},

    // Model control API
    {StatusCode::OK_NOT_RELOADED, "Config reload was not needed"},
    {StatusCode::OK_RELOADED, "Config reload successful"},

    // Metrics
    {StatusCode::INVALID_METRICS_ENDPOINT, "Metrics config endpoint path is invalid"},
    {StatusCode::INVALID_METRICS_FAMILY_NAME, "Invalid name in metrics_list"},
    {StatusCode::METRICS_REST_PORT_MISSING, "Missing rest_port parameter in server CLI"},

    // C-API
    {StatusCode::DOUBLE_BUFFER_SET, "Cannot set buffer more than once to the same tensor"},
    {StatusCode::DOUBLE_TENSOR_INSERT, "Cannot insert more than one tensor with the same name"},
    {StatusCode::DOUBLE_PARAMETER_INSERT, "Cannot insert more than one parameter with the same name"},
    {StatusCode::NONEXISTENT_BUFFER_FOR_REMOVAL, "Tried to remove nonexisting buffer"},
    {StatusCode::NONEXISTENT_PARAMETER, "Tried to use nonexisting parameter"},
    {StatusCode::NONEXISTENT_TENSOR, "Tried to get nonexisting tensor"},
    {StatusCode::NONEXISTENT_TENSOR_FOR_SET_BUFFER, "Tried to set buffer for nonexisting tensor"},
    {StatusCode::NONEXISTENT_TENSOR_FOR_REMOVE_BUFFER, "Tried to remove buffer for nonexisting tensor"},
    {StatusCode::NONEXISTENT_TENSOR_FOR_REMOVAL, "Tried to remove nonexisting tensor"},
    {StatusCode::NONEXISTENT_LOG_LEVEL, "Tried to use nonexisting log level"},
    {StatusCode::NONEXISTENT_PTR, "Tried to use nonexisting pointer"},
    {StatusCode::SERVER_NOT_READY, "Server is not ready"},

    // Server Start errors
    {StatusCode::OPTIONS_USAGE_ERROR, "options validation error"},
    {StatusCode::FAILED_TO_START_GRPC_SERVER, "Failed to start gRPC server"},
    {StatusCode::FAILED_TO_START_REST_SERVER, "Failed to start REST server"},
    {StatusCode::SERVER_ALREADY_STARTED, "Server has already started"},
    {StatusCode::MODULE_ALREADY_INSERTED, "Module already inserted"},
};
}  // namespace ovms
