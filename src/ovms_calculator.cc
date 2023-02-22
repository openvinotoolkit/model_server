// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ovms.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
namespace mediapipe {
namespace tf = ::tensorflow;

// A Calculator that simply passes its input Packets and header through,
// unchanged.  The inputs may be specified by tag or index.  The outputs
// must match the inputs exactly.  Any number of input side packets may
// also be specified.  If output side packets are specified, they must
// match the input side packets exactly and the Calculator passes its
// input side packets through, unchanged.  Otherwise, the input side
// packets will be ignored (allowing PassThroughCalculator to be used to
// test internal behavior).  Any options may be specified and will be
// ignored.
class OVMSCalculator : public CalculatorBase {
    OVMS_Server* cserver {nullptr};
    OVMS_ServerSettings* _serverSettings = nullptr;
    OVMS_ModelsSettings* _modelsSettings = nullptr;
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    std::cout << __FILE__ << " " << __LINE__ << std::endl;
    // inputs contract
    RET_CHECK(!cc->Inputs().GetTags().empty());
    cc->Inputs().Tag("TAG").Set<tf::Tensor>();
    // outputs contract
    RET_CHECK(!cc->Outputs().GetTags().empty());
    cc->Outputs().Tag("TAG").Set<tf::Tensor>();
    // TODO add handling side packet/options for servable name, version
    // TODO check for other tags and return error
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) final {
        // Close is called on input node and output node in initial pipeline
        //OVMS_ServerDelete(cserver);
        OVMS_ModelsSettingsDelete(_modelsSettings);
        OVMS_ServerSettingsDelete(_serverSettings);
    return absl::OkStatus();
  }
  absl::Status Open(CalculatorContext* cc) final {
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).Header().IsEmpty()) {
        cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
      }
    }
    if (cc->OutputSidePackets().NumEntries() != 0) {
      for (CollectionItemId id = cc->InputSidePackets().BeginId();
           id < cc->InputSidePackets().EndId(); ++id) {
        cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
      }
    }
    cc->SetOffset(TimestampDiff(0));
    OVMS_ServerNew(&cserver);
    OVMS_ServerSettingsNew(&_serverSettings);
    OVMS_ModelsSettingsNew(&_modelsSettings);
    OVMS_ModelsSettingsSetConfigPath(_modelsSettings, "/ovms/src/test/c_api/config_standard_dummy.json");
    OVMS_ServerStartFromConfigurationFile(cserver, _serverSettings, _modelsSettings);
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    cc->GetCounter("PassThrough")->Increment();
    if (cc->Inputs().NumEntries() == 0) {
      return tool::StatusStop();
    }
    for (CollectionItemId id = cc->Inputs().BeginId();
         id < cc->Inputs().EndId(); ++id) {
      if (!cc->Inputs().Get(id).IsEmpty()) {
        VLOG(3) << "Passing " << cc->Inputs().Get(id).Name() << " to "
                << cc->Outputs().Get(id).Name() << " at "
                << cc->InputTimestamp().DebugString();
        cc->Outputs().Get(id).AddPacket(cc->Inputs().Get(id).Value());
      }
    }
    OVMS_InferenceRequest* request{nullptr};
    OVMS_InferenceRequestNew(&request, cserver, "dummy", 1);
    const char* DUMMY_MODEL_INPUT_NAME = "b";
    const std::vector<size_t> DUMMY_MODEL_SHAPE {1,10};
    const size_t DUMMY_MODEL_INPUT_SIZE = 10;
    // adding input
    OVMS_InferenceRequestAddInput(request, DUMMY_MODEL_INPUT_NAME, OVMS_DATATYPE_FP32, DUMMY_MODEL_SHAPE.data(), DUMMY_MODEL_SHAPE.size());
    // setting buffer
    std::array<float, DUMMY_MODEL_INPUT_SIZE> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const uint32_t notUsedNum = 0;
    OVMS_InferenceRequestInputSetData(request, DUMMY_MODEL_INPUT_NAME, reinterpret_cast<void*>(data.data()), sizeof(float) * data.size(), OVMS_BUFFERTYPE_CPU, notUsedNum);
    //////////////////
    //  INFERENCE
    //////////////////
    OVMS_InferenceResponse* response = nullptr;
    OVMS_Inference(cserver, request, &response);
    uint32_t outputCount = 42;
    uint32_t parameterCount = 42;

    const void* voutputData;
    size_t bytesize = 42;
    uint32_t outputId = 0;
    OVMS_DataType datatype = (OVMS_DataType)199;
    const uint64_t* shape{nullptr};
    uint32_t dimCount = 42;
    OVMS_BufferType bufferType = (OVMS_BufferType)199;
    uint32_t deviceId = 42;
    const char* outputName{nullptr};
    // verify GetOutputCount
    OVMS_InferenceResponseGetOutputCount(response, &outputCount);
    // verify GetParameterCount
    OVMS_InferenceResponseGetParameterCount(response, &parameterCount);
    // verify GetOutput
    if(OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId) != 0){
        std::cout << "Check config file, ResponseGetOutput error." << std::endl;
    }

    std::cout << std::endl << "shape: ";
    for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
            std::cout << shape[i] << " ";
    }
    std::cout << std::endl << "data: ";
    const float* outputData = reinterpret_cast<const float*>(voutputData);
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << outputData[i] << " ";
    }
    std::cout << std::endl;

    return absl::OkStatus();
  }
};
REGISTER_CALCULATOR(OVMSCalculator);

}  // namespace mediapipe
