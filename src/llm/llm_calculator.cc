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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#pragma GCC diagnostic pop
#include <memory>

#include "src/kfserving_api/grpc_predict_v2.grpc.pb.h"
#include "src/kfserving_api/grpc_predict_v2.pb.h"

#include <openvino/openvino.hpp>
#include <llm_engine.hpp>
#include <scheduler.hpp>
#include <paged_attention.hpp>
#include <model_config.hpp>

using KFSRequest = inference::ModelInferRequest;
using KFSResponse = inference::ModelInferResponse;

constexpr size_t BATCH_SIZE = 1;

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t> tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

std::string detokenize(ov::InferRequest& detokenizer, ov::Tensor tokens) {
    detokenizer.set_input_tensor(tokens);
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}


namespace mediapipe {

class LLMCalculator : public CalculatorBase {
    ov::Core core;
    ov::InferRequest tokenizer, detokenizer, llm;
    std::unique_ptr<LLMEngine> engine;

public:
    static absl::Status GetContract(CalculatorContract* cc) {
        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract start";
        RET_CHECK(!cc->Inputs().GetTags().empty());
        RET_CHECK(!cc->Outputs().GetTags().empty());

        cc->Inputs().Tag("REQUEST").Set<const KFSRequest*>();
        cc->Outputs().Tag("RESPONSE").Set<KFSResponse>();

        LOG(INFO) << "LLMCalculator [Node: " << cc->GetNodeName() << "] GetContract end";
        return absl::OkStatus();
    }

    absl::Status Close(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Close";
        return absl::OkStatus();
    }

    absl::Status Open(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open start";
        ov::Core core;
        core.add_extension("/openvino_tokenizers/build/src/libopenvino_tokenizers.so");
        core.add_extension<PagedAttention>();

        tokenizer = core.compile_model("/workspace/openvino_tokenizer.xml", "CPU").create_infer_request();
        detokenizer = core.compile_model("/workspace/openvino_detokenizer.xml", "CPU").create_infer_request();

        // The model can be compiled for GPU as well
        std::shared_ptr<ov::Model> model = core.read_model("/workspace/vllm_optimum_openvino_model.xml");
        // TODO: reshape model according to plugin desired shape condifuration
        const ov::ParameterVector& parameters = model->get_parameters();
        ov::PartialShape pshape = ov::PartialShape::dynamic(4);
        for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
            parameters[2 + 2 * decoder_layer_id]->set_element_type(kv_cache_precision);
            parameters[2 + 2 * decoder_layer_id + 1]->set_element_type(kv_cache_precision);
            parameters[2 + 2 * decoder_layer_id]->set_partial_shape(pshape);
            parameters[2 + 2 * decoder_layer_id + 1]->set_partial_shape(pshape);
        }
        model->validate_nodes_and_infer_types();

        SchedulerConfig scheduler_config {
            .max_num_batched_tokens = 256,
            .num_kv_blocks = NUM_BLOCKS,
            .dynamic_split_fuse = false,
            .max_num_seqs = 256, // not used if dynamic_split_fuse=True
            .max_paddings = 256, // not used if dynamic_split_fuse=True
        };

        llm = core.compile_model(model, "CPU", ov::enable_profiling(true), ov::hint::enable_hyper_threading(false)).create_infer_request();
        engine = std::make_unique<LLMEngine>(llm, scheduler_config);

        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Open end";
        return absl::OkStatus();
    }

#define RETURN_EXECUTION_FAILED_STATUS() \
    return absl::Status(absl::StatusCode::kInternal, "Error occurred during graph execution")

    absl::Status Process(CalculatorContext* cc) final {
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process start";
       try {
            const KFSRequest *request = cc->Inputs().Tag("REQUEST").Get<const KFSRequest*>();
            // Hardcoded single input for data
            auto data = request->raw_input_contents().Get(0);
            std::string prompt = std::string(data.begin(), data.end());
            LOG(INFO) << "Received prompt: " << prompt;

            /*
            TODO: 
            
            Real work here...
            
            */
            auto [_input_ids_prompt, _attention_mask_prompt] = tokenize(tokenizer, prompt);
            SamplingParameters greedy_search = SamplingParameters::greedy();
            greedy_search.max_new_tokens = 256;
            engine->add_request(1, _input_ids_prompt, greedy_search);

            std::string resultStr;

            for (size_t num_finished = 0; engine->has_running_requests(); ) {
                LOG(INFO) << "Running engine step";
                std::vector<GenerationResult> results = engine->step();
                if (!results.empty()) {
                    num_finished += results.size();
                    std::cout << "Output tokens: ";
                    for (size_t i = 0; i < results[0].m_generation_ids[0].size(); ++i) {
                        std::cout << results[0].m_generation_ids[0][i] << ", ";
                    }
                    resultStr = detokenize(detokenizer, results[0].m_generation_ids[0]);
                    std::cout << std::endl << "Output string: " << resultStr << std::endl;
                    LOG(INFO) << "Finished: " << num_finished;
                }
            }

            //--------------------------------------------
            
            std::string outputStr = resultStr; //"Hardcoded output";

            auto response = std::make_unique<KFSResponse>();
            auto* responseOutput = response->add_outputs();
            responseOutput->set_name("output");
            responseOutput->set_datatype("BYTES");
            responseOutput->clear_shape();
            responseOutput->add_shape(outputStr.size());
            response->add_raw_output_contents()->assign(reinterpret_cast<char*>(outputStr.data()), outputStr.size());

            cc->Outputs().Tag("RESPONSE").AddPacket(MakePacket<KFSResponse>(*response).At(cc->InputTimestamp()));


        } catch (std::exception& e) {
            LOG(INFO) << "Error occurred during node " << cc->NodeName() << " execution: " << e.what();
            RETURN_EXECUTION_FAILED_STATUS();
        } catch (...) {
            LOG(INFO) << "Unexpected error occurred during node " << cc->NodeName() << " execution";
            RETURN_EXECUTION_FAILED_STATUS();
        }
        LOG(INFO) << "LLMCalculator [Node: " << cc->NodeName() << "] Process end";
        return absl::OkStatus();
    }
};

REGISTER_CALCULATOR(LLMCalculator);
}  // namespace mediapipe
