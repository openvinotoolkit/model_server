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

import pickle
from functools import reduce
from pathlib import Path

import numpy as np

from common_libs.assertions import CapiException
from tests.functional.data.ovms_capi_wrapper import ovms_capi_shared

cimport ovms_capi as capi  # `ovms_capi.pxd`
from libc.stdint cimport int64_t, uint8_t, uint32_t, uint64_t, uintptr_t
from libc.stdio cimport printf
from libc.stdlib cimport malloc


cdef capi.OVMS_Server * srv = NULL
cdef capi.OVMS_ServerSettings * serverSettings = NULL
cdef capi.OVMS_ModelsSettings * modelsSettings = NULL

LOG_LEVELS = {
    "TRACE": capi.OVMS_LogLevel_enum.OVMS_LOG_TRACE,
    "DEBUG": capi.OVMS_LogLevel_enum.OVMS_LOG_DEBUG,
    "INFO": capi.OVMS_LogLevel_enum.OVMS_LOG_INFO,
    "WARNING": capi.OVMS_LogLevel_enum.OVMS_LOG_WARNING,
    "ERROR": capi.OVMS_LogLevel_enum.OVMS_LOG_ERROR,
}

OVMS_DATATYPE_TO_NUMPY = {
    capi.OVMS_DATATYPE_FP32: np.float32,
    capi.OVMS_DATATYPE_I32: np.int32,
    capi.OVMS_DATATYPE_I64: np.int64,
    capi.OVMS_DATATYPE_U8: np.uint8,
}  #TODO: Add remaining types


def pickled_function_parameters(func, **kwargs):
    def wrapper(**kwargs):
        global tmp_infile
        global tmp_outfile
        # Get input parameters as kwargs
        _raw_kwargs = Path(tmp_infile).read_bytes()
        kwargs = pickle.loads(_raw_kwargs)
        print(f"kwargs={kwargs}")
        try:
            result = func(**kwargs)
        except Exception as exc:
            result = str(exc)     # Store exception

        # Store resulting object
        Path(tmp_outfile).write_bytes(pickle.dumps(result))
    return wrapper


def ovms_start_server(parameters):
    cdef uint32_t grpc_port = parameters["grpc_port"] if parameters["grpc_port"] else 0
    cdef uint32_t rest_port = parameters["rest_port"] if parameters["rest_port"] else 0
    log_level = parameters["log_level"]

    config_path = parameters["config_path"].encode()
    cdef char *config_file = <char *> config_path

    ovms_get_capi_version()

    capi.OVMS_ServerSettingsNew(&serverSettings)
    capi.OVMS_ModelsSettingsNew(&modelsSettings)
    capi.OVMS_ServerNew(&srv)

    printf("Starting OVMS CAPI server:\n")
    printf("config_path %s\n", config_file)
    printf("grpc_port %d\n", grpc_port)
    printf("rest_port %d\n", rest_port)
    #print(f"log_level {log_level}")

    cdef char *cpu_ext_file = NULL
    if "cpu_extension_path" in parameters:
        cpu_ext_path = parameters["cpu_extension_path"].encode()
        cpu_ext_file = <char *> cpu_ext_path
        printf("cpu_extension_path %s\n", cpu_ext_file)
        capi.OVMS_ServerSettingsSetCpuExtensionPath(serverSettings, cpu_ext_file)

    _poll_wait_seconds = parameters.get("file_system_poll_wait_seconds", None)
    if _poll_wait_seconds is not None:
        printf("file_system_poll_wait_seconds %d\n", <uint32_t> int(_poll_wait_seconds))
        capi.OVMS_ServerSettingsSetFileSystemPollWaitSeconds(serverSettings, <uint32_t> int(_poll_wait_seconds))

    capi.OVMS_ServerSettingsSetGrpcPort(serverSettings, grpc_port)
    capi.OVMS_ServerSettingsSetRestPort(serverSettings, rest_port)
    capi.OVMS_ServerSettingsSetLogLevel(serverSettings, LOG_LEVELS[log_level])
    capi.OVMS_ModelsSettingsSetConfigPath(modelsSettings, config_file)

    cdef capi.OVMS_Status * res = capi.OVMS_ServerStartFromConfigurationFile(srv, serverSettings, modelsSettings)
    if res:
        # analyze error from OVMS_Status
        printf("OVMS_Status %x\n", <void *> res)
    else:
        printf("Started OVMS Server: %x\n", <void *> res)
    return int(<uintptr_t> srv)

def parse_error(_res):
    cdef uint32_t code = 0
    cdef const char * details = NULL
    cdef capi.OVMS_Status* res = <capi.OVMS_Status*> _res
    capi.OVMS_StatusCode(res, &code)
    capi.OVMS_StatusDetails(res, &details)

    print(f"Error during inference: code {int(code)}, details: {str(details)}\n")
    capi.OVMS_StatusDelete(res)

def server_stop():
    global srv, modelsSettings, serverSettings
    printf("Stopping server: %lX\n", <uintptr_t>srv)
    capi.OVMS_ServerDelete(srv)
    srv = NULL
    capi.OVMS_ModelsSettingsDelete(modelsSettings)
    modelsSettings = NULL
    capi.OVMS_ServerSettingsDelete(serverSettings)
    serverSettings = NULL
    print("Server stopped")

@pickled_function_parameters
def send_inference(model_name, inputs):
    print("Prepare inference")
    cdef capi.OVMS_InferenceRequest* request = NULL
    py_byte_string = model_name.encode("UTF-8")     # little trick
    cdef const char * _name = py_byte_string

    capi.OVMS_InferenceRequestNew(&request,
                                  srv,
                                   _name,   # Model name
                                  1)

    cdef capi.OVMS_DataType _dataType = capi.OVMS_DATATYPE_UNDEFINED
    cdef int64_t * _shape_struct = NULL
    cdef float *_inputData = NULL
    cdef uint32_t dimensions = 0
    cdef size_t _byteSize = 0

    for input_name, _input in inputs.items():
        print(f"Processing input name {input_name}")
        dimensions = <uint32_t> len(_input.shape)
        _shape_struct = <int64_t *> malloc(dimensions * sizeof(int64_t))
        _byteSize =  <size_t> reduce(lambda x,y: x*y, _input.shape) * _input.dtype.itemsize
        _inputData = <float *> malloc(_byteSize)

        if _input.dtype == np.float32:
            _dataType = capi.OVMS_DATATYPE_FP32
        elif _input.dtype == np.uint8:
            _dataType = capi.OVMS_DATATYPE_U8
        elif _input.dtype == np.int32:
            _dataType = capi.OVMS_DATATYPE_I32
        elif _input.dtype == np.int64:
            _dataType = capi.OVMS_DATATYPE_I64
        elif _input.dtype == np.object_:
            _dataType = capi.OVMS_DATATYPE_STRING
        else:
            raise CapiException(f"Unsupported data type: {_input.dtype}")

        py_byte_string = input_name.encode('UTF-8')
        _name = py_byte_string

        for i in range(dimensions):
            _shape_struct[i] = _input.shape[i]

        printf("Input name: %s\n", _name)
        printf("Input dimensions: %u\n", dimensions)
        printf("Input datatype: %u\n", _dataType)
        printf("Input bytesize: %u\n", _byteSize)
        printf("Input shape: %lu %lu %lu %lu\n", _shape_struct[0], _shape_struct[1], _shape_struct[2], _shape_struct[3])


        capi.OVMS_InferenceRequestAddInput(request,
                                           _name,
                                           _dataType,
                                           _shape_struct,
                                           dimensions)

        capi.OVMS_InferenceRequestInputSetData(request,
                                               _name,
                                               _inputData,
                                               _byteSize,
                                               capi.OVMS_BUFFERTYPE_CPU,
                                               0)

    cdef capi.OVMS_InferenceResponse* response = NULL
    cdef capi.OVMS_Status* res = NULL

    print("Sending inference to server: {:x}".format(<uint64_t> srv))
    res = capi.OVMS_Inference(srv, request, &response)
    print("Received inference from server: {:x}".format(<uint64_t> res))

    cdef uint32_t code = 0
    cdef const char * details = NULL
    cdef bytes py_string

    if res:
        capi.OVMS_StatusCode(res, &code)
        capi.OVMS_StatusDetails(res, &details)
        py_string = details
        print(f"Inference failed:\ncode={code}\ndetails={details}")
        capi.OVMS_StatusDelete(res)
        exception = ovms_capi_shared.OvmsInferenceFailed(py_string.decode("UTF-8"))
        exception.code = int(code)
        return exception    # TODO: raise should be more appropriate

    cdef uint32_t outputCount = 0
    capi.OVMS_InferenceResponseOutputCount(response, &outputCount)
    print(f"Inference succeeded, read={outputCount}")
    result = {}

    cdef void* voutputData = NULL
    cdef size_t bytesize = 0
    cdef uint32_t outputId = - 1
    cdef capi.OVMS_DataType datatype = capi.OVMS_DATATYPE_FP32
    cdef int64_t* out_shape = NULL
    cdef size_t dimCount = 0
    cdef capi.OVMS_BufferType bufferType = capi.OVMS_BUFFERTYPE_CPU
    cdef uint32_t deviceId = 42
    cdef const char* outputName = NULL

    for i in range(outputCount):
        outputId = i
        capi.OVMS_InferenceResponseOutput(response,
                                             outputId,
                                             &outputName,
                                             &datatype,
                                             &out_shape,
                                             &dimCount,
                                             &voutputData,
                                             &bytesize,
                                             &bufferType,
                                             &deviceId)

        _shape = [int(out_shape[i]) for i in range(dimCount)]
        _size_to_read = 4 * reduce(lambda x,y: x*y, _shape)
        _output_bytes = bytes([(<uint8_t*> voutputData)[i] for i in range(_size_to_read)])

        print(f"_size_to_read = {_size_to_read}")
        _np_array = np.ndarray(_shape, dtype=np.float32, buffer=_output_bytes)

        printf("OutputId  %u\n", outputId)
        printf("DimCount  %u\n", dimCount)
        printf("Datatype %u\n", datatype)
        printf("BufferType %u\n", bufferType)
        printf("Bytesize %u\n", bytesize)
        print(f"Shape: {_shape}" )
        printf("Output name %s\n", outputName)

        result[outputName.decode("UTF-8")] = _np_array.tolist()

    return result


def ovms_get_capi_version():
    cdef uint32_t major = 0
    cdef uint32_t minor = 0

    capi.OVMS_ApiVersion(&major, &minor)
    printf("C-API version: %x.%x\n", major, minor)

    return int(<unsigned int> major), int(<unsigned int> minor)

@pickled_function_parameters
def get_model_meta(servableName, servableVersion):
    global tmp_outfile
    cdef capi.OVMS_ServableMetadata * servableMetadata = NULL
    cdef char * _servableName = NULL
    cdef int64_t _servableVersion = 0

    py_byte_string = servableName.encode('UTF-8')
    _servableName = py_byte_string
    _servableVersion = 0 if not servableVersion else int(servableVersion)

    printf("Get Model Meta: %s\t%ld", _servableName, _servableVersion)
    capi.OVMS_GetServableMetadata(srv,
                                  _servableName,
                                  _servableVersion,
                                  &servableMetadata)
    printf("ServableMetadata: %x\n", servableMetadata)
    if servableMetadata == NULL:
        # TODO: Raise & handle could be more appropriate
        return  ovms_capi_shared.OvmsModelNotFound("OVMS_GetServableMetadata failed")

    result = {
        "inputs": [],
        "outputs": []
    }

    cdef char *info = NULL
    cdef char *name = NULL
    cdef capi.OVMS_DataType datatype = <capi.OVMS_DataType> 0
    cdef uint32_t id = 0
    cdef size_t dimCount = 0
    cdef uint32_t count = 0
    cdef int64_t *shapeMinArray = NULL
    cdef int64_t *shapeMaxArray = NULL
    capi.OVMS_ServableMetadataInfo(servableMetadata,
                                      <const void **>&info)
    result["info"] = str(info)

    capi.OVMS_ServableMetadataInputCount(servableMetadata, &count)
    result["input_count"] = int(count)
    for i in range(result["input_count"]):
        id = <uint32_t>i
        capi.OVMS_ServableMetadataInput(servableMetadata,
                                           id,
                                           &name,
                                           &datatype,
                                           &dimCount,
                                           &shapeMinArray,
                                           &shapeMaxArray)
        _shape_min = [int(shapeMinArray[i]) for i in range(dimCount)]
        _shape_max = [int(shapeMaxArray[i]) for i in range(dimCount)]
        result["inputs"].append({
            "name": name.decode("UTF-8"),
            "datatype": OVMS_DATATYPE_TO_NUMPY[datatype], # int(datatype),
            "dimCount": int(dimCount),
            "shapeMinArray": _shape_min,
            "shapeMaxArray": _shape_max,
            "shape": _shape_min
        })

    capi.OVMS_ServableMetadataOutputCount(servableMetadata, &count)
    result["output_count"] = int(count)
    for i in range(result["output_count"]):
        id = <uint32_t>i
        capi.OVMS_ServableMetadataOutput(servableMetadata,
                                           id,
                                           &name,
                                           &datatype,
                                           &dimCount,
                                           &shapeMinArray,
                                           &shapeMaxArray)
        _shape_min = [int(shapeMinArray[i]) for i in range(dimCount)]
        _shape_max = [int(shapeMaxArray[i]) for i in range(dimCount)]
        result["outputs"].append({
            "name": name.decode("UTF-8"),
            "datatype": OVMS_DATATYPE_TO_NUMPY[datatype],    #int(datatype),
            "dimCount": int(dimCount),
            "shapeMinArray": _shape_min,
            "shapeMaxArray": _shape_max,
            "shape": _shape_min
        })

    print(f"Meta: {result}")

    Path(tmp_outfile).write_bytes(pickle.dumps(result))

    return result

