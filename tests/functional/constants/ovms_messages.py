#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import re

from tests.functional.utils.environment_info import CurrentOsInfo
from tests.functional.constants.os_type import OsType
from ovms.config import base_os
from tests.functional.constants.ovms import CurrentOvmsType


class OvmsMessages:

    MODEL_VERSION_NOT_FOUND = "Model with requested version is not found"
    MODEL_NAME_OR_VERSION_NOT_FOUND = "Model with requested name and/or version is not found"
    MODEL_VERSION_RETIRED = "Model with requested version is retired"
    MODEL_NAME_VERSION_NOT_FOUND = "Model with requested name and/or version is not found"
    MODEL_INVALID_INPUT_SHAPE = "Invalid input shape"

    MODEL_INVALID_RESHAPED_REQUEST = "Model could not be reshaped with requested shape"
    MODEL_INVALID_NDARRAY = "Could not parse instance content. Not valid ndarray detected"
    MODEL_INVALID_NUMBER_SHAPE_DIMS = "Invalid number of shape dimensions"
    MODEL_UNEXPECTED_INPUT_TENSOR_ALIAS = "Unexpected input tensor alias"
    MODEL_INVALID_BATCH_SIZE = "Wrong batch size parameter provided. Model batch size will be set to default."
    MODEL_BATCH_SIZE_OUT_OF_RANGE = "Wrong batch size parameter provided. Model batch size will be set to default."
    MODEL_INVALID_NIREQ_BIG_VALUE = "error: Nireq parameter too high - Exceeded allowed nireq value"
    MODEL_INVALID_NIREQ_NEGATIVE_VALUE = "error: Nireq parameter cannot be negative value"
    MODEL_INVALID_PRECISION = "Invalid precision"
    MODEL_INVALID_INPUT_PRECISION = "Invalid input precision"
    MODEL_INVALID_INPUT_PRECISION_EXTENDED = "Invalid input precision - Expected: {}; Actual: {}"
    MODEL_MISSING_INPUT_SPECIFIC_NAME = "Missing input with specific name"
    MODEL_RELOADING = "Will reload model: {}; version:"
    MODEL_UNLOADING = "Will unload model: {}; version: {}"
    MODEL_INPUT_NAME_MAPPING_NAME = "Input name: {}; mapping_name: {};"
    MODEL_OUTPUT_NAME_MAPPING_NAME = "Output name: {}; mapping_name: {};"
    MODEL_INPUT_SHAPE_RELOADING = "Input name: {}; mapping_name: {}; shape: {};"
    MODEL_OUTPUT_SHAPE_RELOADING = "Output name: {}; mapping_name: {}; shape: {};"
    MODEL_INPUT_SHAPE_RELOADING_DETAIL = "Input name: {}; mapping_name: {}; shape: {}; precision: {}; layout: {}"
    MODEL_OUTPUT_SHAPE_RELOADING_DETAIL = "Output name: {}; mapping_name: {}; shape: {}; precision: {}; layout: {}"
    MODEL_FAILED_TO_LOAD_INTO_DEVICE = "Cannot load network into target device"
    MODEL_INVALID_INFERENCE_INPUT = "Invalid number of inputs - Expected: {}; Actual: {}"
    MODEL_WITH_REQUESTED_VERSION_NOT_LOADED_YET = "Model with requested version is not loaded yet"
    MODEL_STATUS_CHANGE = (
        'STATUS CHANGE: Version {} of model {} status change. New status: ( "state": "{}", "error_code": "{}" )'
    )

    MODEL_LOADING = "Loading model: {}, version: {}, from path: {}"
    PIPELINE_NOT_LOADED_YET = "Pipeline is not loaded yet"
    PIPELINE_IS_RETIRED = "Pipeline is retired"
    MEDIAPIPE_IS_RETIRED = "Mediapipe is retired"
    MEDIAPIPE_EXECUTION_FAILED = "Mediapipe execution failed"
    OVMS_PIPELINE_VALIDATION_FAILED_MSG = "Validation of pipeline: {} definition failed"
    OVMS_PIPELINE_VALIDATION_FAILED_MSG_MISSING_MODEL = (
        "Validation of pipeline: {} definition failed. Missing model: {};"
    )
    OVMS_PIPELINE_VALIDATION_PASS_MSG = "Pipeline: {} state changed to: AVAILABLE after handling: ValidationPassedEvent"
    TWO_PIPELINES_WITH_THE_SAME_NAME = "Duplicated pipeline names"
    PIPELINE_WITH_THE_SAME_NAME_AS_MODEL_NAME = "Pipeline name: {} is already occupied by model"
    PIPELINE_STATE = "Pipeline: {} state changed to: "
    PIPELINE_STARTED = (
        PIPELINE_STATE + "AVAILABLE "
    )  # adding whitespace to avoid matching state AVAILABLE_REQUIRED_REVALIDATION
    PIPELINE_UNLOADED = PIPELINE_STATE + "RETIRED"
    PIPELINE_MULTIPLE_DEMULTIPLEXERS = "PipelineDefinition: {} has multiple demultiplexers with at least one dynamic."
    PIPELINE_LOADING_PRECONDITION_FAILED = (
        "Pipeline: {} state changed to: LOADING_PRECONDITION_FAILED after handling: ValidationFailedEvent:"
    )
    PIPELINE_INVALID_BATCH_SIZE_USED_WITH_DEMULTIPLEXER = (
        "Validation of pipeline: {} definition failed. Shape mismatch between: dependent node"
    )
    PIPELINE_INVALID_DIMENSION_DEMULTIPLY_GREATER_THAN_BATCH_SIZE = (
        "Validation of pipeline: {} definition failed. Demultiply count: {} "
        "of node: elastic_node does not match tensor first dimension value: {}"
    )
    PIPELINE_INVALID_INPUT_SHAPE = (
        "Invalid input shape - Node: {} input: input Invalid shape - Expected: {}; Actual: {}"
    )
    PIPELINE_REFERS_TO_INCORRECT_LIBRARY = "Pipeline refers to incorrect library"

    OVMS_BATCH_SIZE_WARNING = "[warn] Unexpected batch_size value"
    OVMS_INVALID_BATCH_SIZE = "Invalid batch size - Expected: {}; Actual: {}"
    OVMS_INVALID_INPUT_BATCH_SIZE = "Invalid input batch size - Expected: {}; Actual: {}"
    OVMS_INVALID_INPUT_BATCH_SIZE_SHORT = "Invalid input batch size"

    OVMS_MODEL_LOADED = "Loaded model {}; version: {}; batch size: {}; No of InferRequests:"
    OVMS_MODEL_LOADED_NIREQ = "Loaded model {}; version: {}; batch size: {}; No of InferRequests: {}"
    OVMS_MODEL_LOADED_SHORT = "Loaded model"

    OVMS_ERROR_OCCURED_WHILE_LOADING_MODEL_GENERIC = "Error occurred while loading model:"

    OVMS_ERROR_COULD_NOT_FIND_FILE_FOR_MODEL = "Could not find file for model:"
    OVMS_ERROR_INCORRECT_WEIGHTS_IN_BIN_FILE = "Error: Incorrect weights in bin file!"

    OVMS_MODEL_FAILED_TO_LOAD = (
        "Error occurred while loading model: {}; version: {}; error: Cannot load network into target device"
    )

    OVMS_SERVER_UNLOADED_MSG = (
        'Version {} of model {} status change. New status: ( "state": "END", "error_code": "OK" )'
    )

    OVMS_SERVER_RUNNING_MSG = (
        'Version {} of model {} status change. New status: ( "state": "AVAILABLE", "error_code": "OK" )'
    )

    OVMS_SERVICES_RUNNING_MSG = {"grpc": "Started gRPC server", "rest": "Started REST server"}
    OVMS_STOPPING = "Stopping ovms"

    OVMS_LOG_LEVEL_DEBUG = "log level: DEBUG"
    OVMS_REST_WORKERS = "Will start {} REST workers"
    OVMS_MODEL_CACHE_ENABLED = "Model cache is enabled:"
    OVMS_MODEL_CACHE_WITH_CUSTOM_LOADER = "Model: {} has allow cache set to true while using custom loader"
    OVMS_DEBUG_CLI_PARAMETERS = "CLI parameters passed to ovms server"

    OVMS_PLUGIN_CONFIG = "OVMS set plugin settings key: {}; value: {};"
    OVMS_TARGET_DEVICE = "Model: {}; version: {}; target device: {}"
    COMPILED_MODEL_TARGET_DEVICE = "compiled model: {};  version: {}; target device: {}"
    OV_NUMBER_STREAMS = "Number of OpenVINO streams:"
    OV_NIREQ = "No of InferRequests:"
    PLUGIN_CONFIG_FOR_DEVICE = "Plugin config for device: {}"

    ERROR_DURING_LOADING_INPUT_TENSORS = "Error during loading input tensors"
    ERROR_TENSOR_INVALID_CONTENT_SIZE = (
        "Invalid content size of tensor proto - Expected: {} bytes; Actual: {} bytes; input name: {}"
    )
    ERROR_EMPTY_RAW_INPUT_CONTENT = "Invalid message structure - raw_input_content is empty"
    ERROR_ZERO_DIMENSION_IS_NOT_ALLOWED = "has zero dimension which is not allowed"
    ERROR_FAILED_TO_SET_BATCH_SIZE = "Failed to set batch size to {}. Possible reasons are:"
    ERROR_MODEL_VERSION_POLICY_UNSUPPORTED_KEY = "Model version policy contains unsupported key"
    ERROR_NOT_VALID_JSON = "Error: The file is not valid json"
    ERROR_MODEL_NOT_FOUND = "Model with requested name is not found"
    ERROR_PIPELINE_NOT_FOUND = "Pipeline with requested name is not found"
    ERROR_FAILED_TO_PARSE = "error parsing options: Argument '{}' failed to parse" if OsType.Windows in base_os \
        else "error parsing options: Argument ‘{}’ failed to parse"
    ERROR_WRONG_BATCH_SIZE = "Wrong batch size parameter provided"
    ERROR_FAILED_TO_PARSE_SHAPE_SHORT = "There was an error parsing shape"
    ERROR_FAILED_TO_PARSE_SHAPE = "There was an error parsing shape {}"
    ERROR_VALUE_HAS_TO_BE_GREATER_THAN_0 = "value has to be greater than 0"
    ERROR_STRING_VAL_IS_EMPTY = "String val is empty"
    ERROR_BINARY_DATA_SIZE_INVALID = "binary_data_size parameter is invalid and cannot be parsed"

    DETECTION_OUTPUT_HAS_ZERO_DIMENSION = "DetectionOutput has zero dimension which is not allowed"
    BATCH_SIZE_IS_ZERO = "Batch size is zero"
    DATA_BATCH_NOT_MATCH = "Data batch and filters rank do not match"

    WARNING_NO_VERSION_FOUND_FOR_MODEL = "No version found for model"

    if not CurrentOvmsType.is_none_type():
        ERROR_COUNT_SHOULD_BE_FROM_1 = f"count should be from 1 to CPU core count : {CurrentOsInfo.get_cpu_amount()}"
        ERROR_REST_WORKERS_COUNT_SHOULD_BE_FROM_1 = f"rest workers {ERROR_COUNT_SHOULD_BE_FROM_1}"
        ERROR_GRPC_WORKERS_COUNT_SHOULD_BE_FROM_1 = f"grpc_workers {ERROR_COUNT_SHOULD_BE_FROM_1}"
    ERROR_WORKERS_COUNT_INFORMATION = "workers count should be from 2 to 10000"
    ERROR_SHAPES_INCONSISTENT = "OV does not support reshaping model: {} with provided shape"
    ERROR_WRONG_LAYER_NAME = "Config shape - {} not found in model"
    ERROR_WRONG_SHAPE_FORMAT = "Error: The provided shape is in wrong format"
    ERROR_WRONG_VERSION_POLICY_FORMAT = "Error: Model version policy is in wrong format"
    ERROR_WRONG_VERSION_POLICY_PROPERTY = "Error: Model version policy contains unsupported key"
    ERROR_CANNOT_PARSE_VERSION_POLICY_PROPERTY = "Couldn't parse model version policy"
    ERROR_NOT_REGISTERED_TARGET_DEVICE = 'Device with "{}" name is not registered in the OpenVINO Runtime'

    ERROR_FAILED_TO_LOAD_LIBRARY = "error: Cannot load library"
    ERROR_FAILED_TO_CREATE_PLUGIN = "error: Failed to create plugin"

    ERROR_WRONG_PLUGIN_CONFIG_FORMAT = "Error: Plugin config is in wrong format"
    ERROR_PLUGIN_CONFIG_UNSUPPORTED_PROPERTY = "Plugin config key: {} not found in supported config keys for device:"
    ERROR_INVALID_LOGGING_LEVEL = "log_level should be one of: TRACE, DEBUG, INFO, WARNING, ERROR"
    ERROR_INVALID_JSON = "The file is not valid json"
    ERROR_INVALID_JSON_STRUCTURE = "Invalid JSON structure. Missing instances in row format"
    ERROR_MODEL_DOES_NOT_SUPPORT_BATCH_SIZE = "Model {} does not support setting batch size"
    S3_WRONG_AUTHORIZATION = "Invalid or missing S3 credentials"
    GS_WRONG_AUTHORIZATION = "Invalid or missing GCS credentials"
    AZURE_WRONG_AUTHORIZATION = (
        "Unable to access path: Server failed to authenticate the request. Make sure the "
        "value of Authorization header is formed correctly including the signature."
    )
    ERROR_DIRECTORY_DOES_NOT_EXISTS = "Directory does not exist: {}"
    ERROR_CANNOT_SERVE_ALL_MODELS = "Cannot serve all models"
    ERROR_CANNOT_ANONYMOUS_RESHAPE_MULTIPLE_INPUTS = "Anonymous fixed shape is invalid for models with multiple inputs"
    ERROR_COULD_NOT_PERFORM_RESHAPE = "Model reshape failed"
    ERROR_WRONG_HTTP_METHOD = "Unsupported method"
    ERROR_UNABLE_TO_ACCESS_PATH = "Unable to access path:"
    INVALID_MODEL_PATH = "The provided base path is invalid or doesn't exists"
    MODEL_CANNOT_BE_PARSED = "Model cannot be parsed"
    NETWORK_NOT_READ_MODEL = "[ NETWORK_NOT_READ ] Unable to read the model"
    UNABLE_TO_READ_THE_MODEL = "Unable to read the model: {}"
    CANNOT_OPEN_LIBRARY = "Cannot open library:"
    NOT_FOUND_MODEL_IN_PATH = "File not found or cannot open"
    ERROR_INVALID_URL = "Invalid request URL"
    WRONG_ENDPOINT = "Wrong endpoint"
    BAD_REQUEST = "Bad Request"
    NOT_IMPLEMENTED = "Not Implemented"
    ERROR_NOT_FOUND = "Not Found"
    ERROR_INPUT_DATA_TOO_BIG = "Input data is too big"
    RESOURCE_EXHAUSTED = {"grpc": "Received message larger than max", "rest": "Request-URI Too Large"}
    ABORTED = "Connection aborted"
    OMVS_SERVICE_UNAVAILABLE = "Service unavailable"
    ERROR_MAX_SEQUENCE_REACHED = "Max sequence number has been reached. Could not create new sequence."
    ERROR_COULDNT_CHECK_DIRECTORY = "Couldn't check directory: {}"
    NO_VERSION_FOUND_FOR_MODEL = "No version found for model in path: {}"
    ERROR_FAILED_TO_CONNECT_TO_ANY_PROXY_ENDPOINT = "Failed to connect to any resolved proxy endpoint"
    ERROR_TERMINATE_CALLED = "terminate called after throwing an instance of"
    STD_SYSTEM_ERROR = "std::system_error"

    ERROR_CFG_JSON_SCHEMA = "JSON schema parse error:#{}"
    ERROR_CFG_NOT_VALID = "Configuration file is not a valid JSON file"
    ERROR_CFG_NOT_FOUND = "Configuration file is invalid {}"
    ERROR_CFG_NOT_VALID_FORMAT = "Configuration file is not in valid configuration format"
    ERROR_CFG_KEY_ERROR = "Keyword:{} Key: #{}"
    USE_CONFIG_PATH_OR_MODEL_PATH_WITH_SPARE_MODEL = "Use either config_path or model_path with model_name"
    USE_CONFIG_PATH_WITHOUT_MODEL = "Use config_path or model_path with model_name"
    ERROR_LOADING_MODEL = "Error occurred while loading model: {}"
    ERROR_LOADING_MODEL_INTERNAL_SERVER_ERROR = (
        "Error occurred while loading model: {}; version: {}; error: Internal server error"
    )
    ERROR_RELOADING_MODEL_INTERNAL_SERVER_ERROR = (
        "Error occurred while reloading model: {}; versions; error: Internal server error"
    )
    ERROR_CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE = "Cannot compile model into target device"
    ERROR_CANNOT_COMPILE_MODEL_INTO_TARGET_DEVICE_OUT_OF_MEMORY = (
        "Cannot compile model into target device; error: Failed to allocate "
        "graph: NC_OUT_OF_MEMORY; model: {}; version: {}; device: {}"
    )
    ERROR_EXCEPTION_FROM_SRC_INFERENCE = "Error: Exception from src/inference/src"

    CUSTOM_LOADER_READING_CONFIGURATION = "Reading Custom Loader: {} configuration"
    CUSTOM_LOADER_NOT_FOUND = "Specified custom loader {} not found."
    CUSTOM_LOADER_TO_BE_USED = "Custom Loader to be used : {}"
    CUSTOM_LOADER_LOOKING_FOR_LOADER = "Looking for loader {} in loaders list"
    CUSTOM_LOADER_INVALID_CUSTOM_LOADER_OPTIONS = "Invalid custom loader options"

    STATEFUL_MODEL_REGISTERED = "Model: {}, version: {}, has been successfully registered in sequence cleaner"
    ERROR_SEQUENCE_ID_NOT_EXIST = "Sequence with provided ID does not exist"
    DEFAULT_MODEL_VERSION_CHANGED = "Updating default version for model: {}, from: {}"

    ERROR_LOADING_PRECONDITION_FAILED = "LOADING_PRECONDITION_FAILED after handling: ValidationFailedEvent"
    ERROR_INVALID_LAYOUT_NHWC = "Received binary image input but resource not configured to accept NHWC layout"
    ERROR_BINARY_INPUTS_CONVERSION_FAILED = "Binary inputs conversion failed."
    ERROR_BINARY_INPUTS_NATIVE_FILE_FORMAT_CONVERSION = "Input native file format conversion failed"
    ERROR_BINARY_INPUTS_CORRUPT_JPEG_DATA = "Corrupt JPEG data"

    ERROR_INTERPOLATE_NODE_IS_NOT_SUPPORTED = "Interpolate node is not supported:"

    ERROR_EXCEPTION_CATCH = "Exception catch:"

    CPU_EXTENSION_LOADING_CUSTOM_CPU_EXT = "Loading custom CPU extension from {}"
    CPU_EXTENSION_LOADED = "Custom CPU extention loaded. Adding it."
    CPU_EXTENSION_ADDED = "Extension added."

    ERROR_CPU_EXTENSION_WILL_NOW_TERMINATE = "- will now terminate."
    ERROR_CPU_EXTENSION_LOADING_FAILED = "Custom CPU extension loading has failed! Reason: {}"

    ERROR_CPU_EXTENSION_FILE_NOT_EXISTS = (
        "File path provided as an --cpu_extension parameter does not exist in the filesystem: {}"
    )

    ERROR_COULDNT_START_MODEL_MANAGER = "Couldn't start model manager"

    NGINX_STARTED_WITH_PID = "Nginx PID: "

    ERROR_METRICS_REST_PORT_MISSING_CFG = {
        False: "rest_port setting is missing, metrics are enabled on rest port",
        True: "Error: Missing rest_port parameter in server CLI",
    }
    ERROR_METRICS_NOT_ENABLED = "metrics_enable setting is missing, required when metrics_list is provided"
    ERROR_METRICS_UNABLE_TO_LOAD_SETTINGS = "Couldn't load metrics settings"
    ERROR_METRICS_INVALID_METRICS = "Error: Invalid name in metrics_list"
    ERROR_METRICS_NOT_SUPPORTED_METRICS = "Metrics family name not supported: {}"

    OPERATOR_UNKNOWN_FIELD = 'strict decoding error: unknown field "{}"'
    OPERATOR_WARNING_UNKNOWN_FIELD = 'Warning: unknown field "spec.models_settings.model_config"'

    TRACE_CHECK_CHANGE = "Checking if something changed with model versions"
    TRACE_CURRENTLY_REGISTERED_MODEL = "Currently registered model: {}"
    OVMS_LOG_LEVEL_TRACE = "log level: TRACE"
    OVMS_LOG_LEVEL = "logLevel={}"

    OVMS_IGNORED_BATCH = "Both shape and batch size have been defined. Batch size parameter will be ignored"

    UNKNOWN_ERROR = "Unknown error"
    FUNCTIONALITY_NOT_IMPLEMENTED = "Functionality not implemented"
    DEMULTIPLICATION_OF_STRINGS_UNSUPPORTED = "Demultiplication of strings in unsupported"

    # CAPI:
    SERVABLE_MANAGER_MODULE_STARTED = "ServableManagerModule started"
    CLEANER_THREAD_STARTED = "Started cleaner thread"
    MODEL_MANAGER_STARTED = "Started model manager thread"
    SERVABLE_MANAGER_MODULE_SHUTDOWN = "ServableManagerModule shutdown"
    CLEANER_THREAD_SHUTDOWN = "Shutdown cleaner thread"
    MODEL_MANAGER_SHUTDOWN = "Shutdown model manager"

    CAPI_STARTING_OVMS_SERVER = "Starting OVMS CAPI server"
    CAPI_STARTED_OVMS_SERVER = "Started OVMS Server:"

    CYTHON_LIBRARY_INIT_MSG = "[1/1] Cythonizing {}"
    CAPI_VERSION = "C-API version: {}.{}"

    CAPI_MODEL_REQUEST = "Processing C-API inference request for servable: {}; version: {}"
    CAPI_TENSOR_ADDED = "Successfully added tensor: {}"
    CAPI_TOTAL_REQUEST_PROCESSING_TIME = "Total C-API req processing time:"

    MEDIAPIPE_GRAPH_DEFINITION_FAILED = "Trying to parse mediapipe graph definition: {} failed"
    MEDIAPIPE_ERROR_PARSING_GRAPH_NO_FIELD = (
        'Message type "mediapipe.CalculatorGraphConfig.Node" has no field named "{}"'
    )
    MEDIAPIPE_FAILED_TO_OPEN_GRAPH_SHORT = "Failed to open mediapipe graph definition:"
    MEDIAPIPE_FAILED_TO_OPEN_GRAPH = "Failed to open mediapipe graph definition: {}, file: {}"
    MEDIAPIPE_PIPELINE_VALIDATION_PASS_MSG = (
        "Mediapipe: {} state changed to: AVAILABLE after handling: ValidationPassedEvent"
    )
    MEDIAPIPE_PIPELINE_VALIDATION_PASS_MSG_SHORT = "state changed to: AVAILABLE after handling: ValidationPassedEvent"
    MEDIAPIPE_UNLOADED = "Mediapipe: {} state changed to: RETIRED"
    MEDIAPIPE_LOADING_PRECONDITION_FAILED = (
        "Mediapipe: {} state changed to: LOADING_PRECONDITION_FAILED after handling: ValidationFailedEvent:"
    )
    LOADING_MEDIAPIPE_SUBCONFIG_FAILED = "Loading Mediapipe {} models from subconfig {} failed."
    MEDIAPIPE_OCCUPIED = "Mediapipe graph name: {} is already occupied by model or pipeline"
    MEDIAPIPE_NO_SUPPORT_FOR_TFLITETENSOR_SERIALIZATION = (
        "There is no support for TfLiteTensor deserialization & serialization"
    )
    MEDIAPIPE_BASE_PATH_NOT_DEFINED = (
        "base_path not defined in config so it will be set to default based on main config directory"
    )
    MEDIAPIPE_GRAPH_PATH_NOT_DEFINED = (
        "graph_path not defined in config so it will be set to default based on base_path and graph name"
    )
    MEDIAPIPE_SUBCONFIG_DEFAULT = "subconfig.json provided for graph: {}"
    MEDIAPIPE_INTERNAL_ERROR_FAILED = (
        "Mediapipe execution failed. MP status - INTERNAL: CalculatorGraph::Run() failed:"
    )
    MEDIAPIPE_NOT_SUPPORTED_PRECISION_FOR_MP = "Not supported precision for Mediapipe tensor deserialization"
    MEDIAPIPE_IS_NOT_LOADED_YET = "Mediapipe is not loaded yet"
    MEDIAPIPE_GRAPH_NOT_FOUND = "Mediapipe graph definition with requested name is not found"
    MEDIAPIPE_ERROR_PARSING_TEXT_FORMAT = "Error parsing text-format mediapipe.CalculatorGraphConfig:"
    MEDIAPIPE_EMPTY_GRAPH = "Trying to parse empty mediapipe graph definition: {} failed"
    MEDIAPIPE_FAILED_TO_ADD_PACKET = "Failed to add packet to mediapipe graph input stream"
    MEDIAPIPE_UNABLE_TO_ATTACH_OBSERVER_TO_OUTPUT_STREAM = 'Unable to attach observer to output stream "{}"'
    MEDIAPIPE_UNEXPECTED_INPUT = "Unexpected input - {} is unexpected"
    MEDIAPIPE_FAILED_TO_LOAD_MEDIAPIPE_GRAPH_DEFINITION = "Failed to open mediapipe graph definition"
    MEDIAPIPE_UNEXPECTED_INPUT_NAME = "Unexpected input name"
    MEDIAPIPE_CALCULATOR_FAILED_TO_LOAD_MODEL = "OpenVINOModelServerSessionCalculator failed to load the model"
    MEDIAPIPE_MESSAGE_CONTAINS_OV_GENAI_IMAGE_TAG = "Message contains restricted <ov_genai_image> tag"
    MEDIAPIPE_REQUEST_PROCESSING_FAILED = "Request processing failed, check its correctness"
    MEDIAPIPE_URL_IMAGE_INVALID_ARGUMENT_FILESYSTEM_DISABLED = "Loading images from local filesystem is disabled"
    MEDIAPIPE_URL_IMAGE_PARSING_FAILED = "Image parsing failed"
    MEDIAPIPE_CANT_OPEN_URL_IMAGE = "parsing failed: can\\'t fopen"
    MEDIAPIPE_UNKNOWN_IMAGE_TYPE = "parsing failed: unknown image type"
    MEDIAPIPE_URL_DOES_NOT_MATCH_ALLOWED_DOMAIN = \
        "Given url does not match any allowed domain from allowed_media_domains"

    PYTHON_NODE_HANDLER_PATH_NOT_EXISTS = "Python node handler_path: {} does not exist."
    PYTHON_NODE_FAILED_TO_PROCESS_GRAPH = "Failed to process python node graph python_model"
    PYTHON_NODE_FAILED_TO_EXECUTE_GRAPH = "Failed to execute mediapipe graph: python_model since it is not available"
    PYTHON_EXECUTOR_MISSING_REQUIRED_FIELDS = (
        'Value of type "mediapipe.PythonExecutorCalculatorOptions" stored in google.protobuf.Any has missing '
        "required fields"
    )
    PYTHON_NODE_FAILED_TO_PROCESS_CUSTOM_NODE_FILE = "Failed to process python node file {} : {}"

    PYTHON_NODE_ERROR_DURING_GRAPH_EXECUTION = (
        'Calculator::Process() for node "{}" failed: Error occurred during graph execution'
    )
    PYTHON_NODE_ERROR_DURING_NODE_EXECUTION = "Error occurred during node {} execution: {}"
    PYTHON_NODE_ERROR_DURING_INITIALIZATION = "Error during python node initialization for handler_path: {} - {}"
    PYTHON_NODE_ERROR_DURING_INITIALIZATION_EMPTY_MODULE_NAME = (
        "Error during python node initialization for handler_path:  - ValueError: Empty module name"
    )
    PYTHON_NODE_ERROR_DURING_INITIALIZATION_EXECUTE_METHOD = (
        "Error during python node initialization. "
        "OvmsPythonModel class defined in {} does not implement execute method."
    )
    PYTHON_NODE_FAILED_TO_PROCESS_FINALIZE_METHOD = "Failed to process python node finalize method. {}"
    PYTHON_NODE_NAME_MISSING_IN_GRAPH = "Python node name is missing in graph: {}"
    PYTHON_MEDIAPIPE_FAILED = "Mediapipe graph: {} initialization failed with message: {}"
    PYTHON_INPUT_SIDE_PACKET_FAILED = "For input side packets ValidatePacketTypeSet failed"
    PYTHON_FAILED_TO_GET_TAG = 'Failed to get tag "PYTHON_NODE_RESOURCES"'
    PYTHON_INPUTS_EMPTY = "!cc->Inputs().GetTags().empty()"
    PYTHON_OUTPUTS_EMPTY = "!cc->Outputs().GetTags().empty()"
    PYTHON_SETTING_INPUT_STREAM = "setting input stream: {} packet type: OVMS_PY_TENSOR from: {}:{}"
    PYTHON_SETTING_OUTPUT_STREAM = "setting output stream: {} packet type: OVMS_PY_TENSOR from: {}:{}"
    PYTHON_INVALID_INPUT_STREAM = (
        'Input Stream "{}" for node with sorted index 0 name upper_text does not have a corresponding output stream'
    )
    PYTHON_INVALID_OUTPUT_STREAM = (
        'Output Stream "{}" for node with sorted index 0 name upper_text does not have a corresponding output stream'
    )
    PYTHON_OVMS_PY_TENSOR_REASSIGNED = 'tag "{}" index 0 already had a name "{}" but is being reassigned a name "{}"'
    PYTHON_DONE_EXECUTION = "Graph {}: Done execution"
    PACKET_TIMESTAMP_MISMATCH = 'Packet timestamp mismatch on a calculator receiving from stream "{}".'
    PACKET_TIMESTAMP_MISMATCH_SHORT = "Packet timestamp mismatch on a calculator"
    PACKET_TIMESTAMP_EXPECTED = "Current minimum expected timestamp is {} but received {}."
    PACKET_TIMESTAMP_INPUT_STREAM_HANDLER = (
        "Are you using a custom InputStreamHandler? Note that some InputStreamHandlers allow timestamps that are not "
        "strictly monotonically increasing. See for example the ImmediateInputStreamHandler class comment."
    )
    PYTHON_NODE_RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED: CalculatorGraph::Run() failed:"
    PYTHON_NODE_ERROR_ALREADY_PROCESSING_DATA = (
        'Calculator::Process() for node "{}" failed: Node is already processing data. '
        'Create new stream for another request.'
    )

    CHAT_TEMPLATE_NOT_LOADED = (
        "Warning: Chat template has not been loaded properly. Servable will not respond to /chat/completions endpoint"
    )
    CHAT_TEMPLATE_LOADING_FAILED = "Chat template loading failed with an unexpected error"  # only in DEBUG mode
    CHAT_TEMPLATE_SYNTAX_ERROR = "Chat template loading failed with error: TemplateSyntaxError"
    CHAT_TEMPLATE_TYPE_ERROR = "Chat template loading failed with error: TypeError"
    CHAT_TEMPLATE_CALCULATOR_ERROR = (
        'Calculator::Process() for node "LLMExecutor" failed: '
        'Error: Chat template not loaded correctly, so it cannot be applied'
    )
    CHAT_TEMPLATE_CALCULATOR_ERROR_SHORT = 'Calculator::Process() for node "LLMExecutor" failed'
    LLM_REQUEST_PROCESSING_FAILED = "Request processing failed, check its correctness."
    LLM_MAX_LENGTH_MUST_BE_GREATER_THAN_PROMPT_TOKENS = \
        r"\'max_length\' must be greater than the number of prompt tokens"
    LLM_ALL_REQUESTS_SCHEDULED_REQUESTS = "All requests: {}; Scheduled requests: {};"
    LLM_ALL_REQUESTS_SCHEDULED_REQUESTS_NPU = "All requests: {};"
    LLM_ALL_REQUESTS_SCHEDULED_REQUESTS_CACHE_USAGE = \
        "All requests: {}; Scheduled requests: {}; Cache type: static, cache usage: {}%"
    LLM_CACHE_USAGE = "Cache usage {}%;"
    LLM_CALCULATOR_CLOSE = "LLMCalculator [Node: LLMExecutor ] Close"
    LLM_STREAM_GENERATION_CANCELLED = "graph wait until done CANCELLED"
    LLM_UNARY_GENERATION_CANCELLED = "Mediapipe execution failed. MP status - CANCELLED"
    LLM_STREAM_GENERATION_DONE = "data: [DONE]"
    LLM_UNARY_GENERATION_DONE = "Complete unary response"
    MEDIAPIPE_SINGLE_MESSAGE_ONLY = "This servable accepts only single message requests"
    PROMPT_IS_NOT_A_STRING = "prompt is not a string"
    ERROR_TYPE_MUST_BE_STRING_BUT_ITS_OBJECT = "type must be string, but is object"
    ERROR_TYPE_MUST_BE_STRING_BUT_ITS_NUMBER = "type must be string, but is number"
    FINAL_PROMPT_EMPTY = "Final prompt after applying chat template is empty"
    ERROR_UNABLE_TO_CAST_PYTHON_INSTANCE = "Chat template loading failed with error: Unable to cast Python instance"
    ERROR_UNSUPPORTED_CHAT_TEMPLATE_FORMAT = "Unsupported chat_template format in file"

    EMPTY_RESPONSE_CONTENT = "Empty response content"
    FINISH_REASON_NULL = '"finish_reason":null'

    GENERAL_ERROR = "[error]"
    HF_IMPORT_MODEL_DOWNLOADED = "Model: {} downloaded"
    HF_IMPORT_GRAPH_CREATED = "Graph: graph.pbtxt created"

    CONFIG_MANIPULATION_ERASING_MODEL = "Erasing model from config: {}"
    CONFIG_MANIPULATION_MODEL_TO_BE_ADDED = "Model to be added to configuration file"
    CONFIG_MANIPULATION_MODEL_TO_BE_REMOVED_FOUND = "Model to be removed found in configuration file"
    CONFIG_MANIPULATION_CONFIG_UPDATED = "Config updated: {}"

    MODELS_ENDPOINT_MODEL_NOT_FOUND = "Model not found"

    PULL_ERROR_FAILURE_WHEN_RECEIVING_DATA_FROM_PEER = \
        "[ERROR] curl_easy_perform() failed: Failure when receiving data from the peer"
    PULL_ERROR_COULD_NOT_RESOLVE_PROXY_NAME = "[ERROR] curl_easy_perform() failed: Could not resolve proxy name"
    PULL_ERROR_TIMEOUT_WAS_REACHED = "[ERROR] curl_easy_perform() failed: Timeout was reached"
    PULL_ERROR_COULD_NOT_CONNECT_TO_SERVER = "[ERROR] curl_easy_perform() failed: Couldn't connect to server"
    PULL_ERROR_LFS_DOWNLOAD_FAILED = "[ERROR] LFS download failed"
    PULL_STATUS_UNCLEAN = "Status: Unclean status detected in libgit2 repository path"
    PULL_STATUS_FAILED = "Status: Failed in libgit2 execution of status method"
    PULL_INFO_RESUME_DOWNLOAD = "[INFO] curl_easy_perform() trying to resume file download"

    # Audio endpoints
    AUDIO_NOT_VALID_WAV_NOR_MP3 = "Received input file is not valid wav nor mp3 audio file"
    AUDIO_FILE_PARSING_FAILS = "File parsing fails"
    AUDIO_INVALID_LANGUAGE_CODE = "Invalid language code."
    AUDIO_TEMPERATURE_OUT_OF_RANGE = "Temperature out of range(0.0, 2.0)"
    AUDIO_INVALID_TIMESTAMP_GRANULARITIES = 'Invalid timestamp_granularities type. Allowed types: "segment", "word"'
    AUDIO_VOICE_NOT_AVAILABLE = "Requested voice not available"
    AUDIO_STREAMING_NOT_SUPPORTED = "streaming is not supported"
    AUDIO_WORD_TIMESTAMPS_NOT_SUPPORTED = "Word timestamps not supported for this model"


class OvmsMessagesRegex:
    # Regex for capturing OVMS log message timestamp in format "[%Y-%m-%d %H:%M:%S.%f]"
    # ie.: [2021-11-23 12:10:20.155][1][serving][info][server.cpp:106] OpenVINO Model Server 2022.1.e866798
    OVMV_LOG_TIMESTAMP_RE = re.compile(r"\[(\d{4}\-\d+?\-\d+? \d+?:\d+?:\d+?.\d+?)\]")

    STATUS_CHANGE_RE = re.compile("STATUS CHANGE:")


class OvmsTestCapiCythonMessages:
    OVMS_CAPI_STARTED_SERVER = "Started OVMS Server:"
    OVMS_CAPI_STARTED_STOPPED = "Server stopped"
