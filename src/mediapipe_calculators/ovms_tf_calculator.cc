//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include <iostream>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "ovms.h"  // NOLINT
namespace mediapipe {
namespace tf = ::tensorflow;

constexpr char* OVMSTFTensorTag = "TFTENSOR";
using std::cout;
using std::endl;
// A Calculator that simply passes its input Packets and header through,
// unchanged.  The inputs may be specified by tag or index.  The outputs
// must match the inputs exactly.  Any number of input side packets may
// also be specified.  If output side packets are specified, they must
// match the input side packets exactly and the Calculator passes its
// input side packets through, unchanged.  Otherwise, the input side
// packets will be ignored (allowing PassThroughCalculator to be used to
// test internal behavior).  Any options may be specified and will be
// ignored.

const char* MODEL_NAME = "dummy";
const int MODEL_VERSION = 1;

class OVMSCalculator : public CalculatorBase {
    OVMS_Server* cserver{nullptr};
    OVMS_ServerSettings* _serverSettings = nullptr;
    OVMS_ModelsSettings* _modelsSettings = nullptr;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        std::cout << __FILE__ << " " << __LINE__ << std::endl;
        // inputs contract
        RET_CHECK(!cc->Inputs().GetTags().empty());
        cc->Inputs().Tag(OVMSTFTensorTag).Set<tf::Tensor>();
        // outputs contract
        RET_CHECK(!cc->Outputs().GetTags().empty());
        cc->Outputs().Tag(OVMSTFTensorTag).Set<tf::Tensor>();
        // TODO add handling side packet/options for servable name, version
        // TODO check for other tags and return error
        cout << __FILE__ << ":" << __LINE__ << endl;
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        cout << __FILE__ << ":" << __LINE__ << endl;
        // Close is called on input node and output node in initial pipeline
        // Commented out since for now this happens twice in 2 nodes graph. Server will close
        // anyway with application closuer TODO fix before release
        // OVMS_ServerDelete(cserver);
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
        OVMS_ServerSettingsSetLogLevel(_serverSettings, OVMS_LOG_DEBUG);
        OVMS_ServerStartFromConfigurationFile(cserver, _serverSettings, _modelsSettings);
        cout << __FILE__ << ":" << __LINE__ << endl;
        return absl::OkStatus();
    }

    absl::Status Process(CalculatorContext* cc) final {
        cout << __FILE__ << ":" << __LINE__ << endl;
        if (cc->Inputs().NumEntries() == 0) {
            return tool::StatusStop();
        }
        std::cout << __FILE__ << " " << __LINE__ << std::endl;
        // extract packet
        cout << __FILE__ << ":" << __LINE__ << endl;
        // extract single tensor
        auto& packets = cc->Inputs().Tag(OVMSTFTensorTag).Get<tf::Tensor>();
        // extract single tensor
        tf::Tensor input_tensor(packets);
        auto input_tensor_access = input_tensor.tensor<float, 2>();  // 2 since dummy is 2d
        cout << endl
             << "Calculator received tensor: [ ";
        for (int x = 0; x < 10; ++x) {
            cout << input_tensor_access(0, x) << " ";
        }
        cout << " ]" << endl;
        tensorflow::serving::PredictRequest tfsrequest;
        tensorflow::serving::PredictResponse tfsresponse;
        tfsrequest.mutable_model_spec()->mutable_name()->assign(MODEL_NAME);
        tfsrequest.mutable_model_spec()->mutable_version()->set_value(MODEL_VERSION);
        const char* DUMMY_MODEL_INPUT_NAME = "b";
        const std::vector<int64_t> DUMMY_MODEL_SHAPE{1, 10};
        const size_t DUMMY_MODEL_INPUT_SIZE = 10;

        cout << __FILE__ << ":" << __LINE__ << endl;
        // TODO check retcode
        input_tensor.AsProtoTensorContent(&(*tfsrequest.mutable_inputs())[DUMMY_MODEL_INPUT_NAME]);
        auto sth = OVMS_GRPCInference((void*)&tfsrequest, (void*)&tfsresponse);
        if (sth != nullptr) {
            cout << "Sth nonnulptr " << endl;
        }
        // here we may need to add additional include from within OVMS to expose prediction service
        // tf::TensorProto* input_tensor_proto = new  tf::TensorProto;
        cout << __FILE__ << ":" << __LINE__ << endl;
        // input_tensor.AsProtoTensorContent(input_tensor_proto);
        cout << __FILE__ << ":" << __LINE__ << endl;
        // TODO construct request
        // TODO receive tf::TensorProto from response
        // TODO ownership of data - what is
        // now we simulate the other way arround
        tf::Tensor output_tensor;
        cout << __FILE__ << ":" << __LINE__ << endl;
        const char* DUMMY_MODEL_OUTPUT_NAME = "a";
        // here we have TF tensor proto
        auto output = tfsresponse.outputs().at(DUMMY_MODEL_OUTPUT_NAME);
        // here we have TF tensor
        output_tensor.FromProto(output);
        // TODO why 0, handling timestamps
        CollectionItemId id = cc->Inputs().BeginId();
        auto outputPacketContent = std::make_unique<tf::Tensor>(output_tensor);
        cc->Outputs().Tag(OVMSTFTensorTag).Add(outputPacketContent.release(), cc->InputTimestamp());

        OVMS_InferenceRequest* request{nullptr};
        OVMS_InferenceRequestNew(&request, cserver, MODEL_NAME, MODEL_VERSION);
        // adding input
        cout << __FILE__ << ":" << __LINE__ << endl;
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
        const int64_t* shape{nullptr};
        size_t dimCount = 42;
        OVMS_BufferType bufferType = (OVMS_BufferType)199;
        uint32_t deviceId = 42;
        const char* outputName{nullptr};
        // verify GetOutputCount
        OVMS_InferenceResponseGetOutputCount(response, &outputCount);
        // verify GetParameterCount
        OVMS_InferenceResponseGetParameterCount(response, &parameterCount);
        // verify GetOutput
        if (OVMS_InferenceResponseGetOutput(response, outputId, &outputName, &datatype, &shape, &dimCount, &voutputData, &bytesize, &bufferType, &deviceId) != 0) {
            std::cout << "Check config file, ResponseGetOutput error." << std::endl;
        }

        std::cout << std::endl
                  << __FILE__ << ":" << __LINE__ << " shape: ";
        for (size_t i = 0; i < DUMMY_MODEL_SHAPE.size(); ++i) {
            std::cout << shape[i] << " ";
        }
        std::cout << std::endl
                  << __FILE__ << ":" << __LINE__ << " data: ";
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
