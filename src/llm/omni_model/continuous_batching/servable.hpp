#pragma once

#include "../../visual_language_model/continuous_batching/servable.hpp"

namespace ovms {

class OmniModelServable : public VisualLanguageModelServable {
public:
    OmniModelServable() {
        properties->inputProcessorContext.config.audioInputConfig = AudioInputConfig{};
    }

    absl::Status validateEndpoint(Endpoint endpoint) const override {
        if (endpoint == Endpoint::COMPLETIONS) {
            return absl::InvalidArgumentError("Omni Servable does not support the /completions endpoint. Use /chat/completions or /responses.");
        }
        if (endpoint == Endpoint::TRANSCRIPTIONS) {
            return absl::InvalidArgumentError("Omni Servable does not support audio transcriptions");
        }
        return absl::OkStatus();
    }

    absl::Status parseRequest(std::shared_ptr<GenAiServableExecutionContext>& executionContext) override {
        auto status = GenAiServable::parseRequest(executionContext);
        if (!status.ok()) {
            return status;
        }
        if (executionContext->apiHandler->getRequest().audioOutputRequested) {
            return absl::InvalidArgumentError("Audio output is not supported by OMNI_CB. Use the OMNI pipeline for audio output.");
        }
        return absl::OkStatus();
    }
};

}  // namespace ovms