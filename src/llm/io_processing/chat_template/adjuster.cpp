//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "adjuster.hpp"

#include "adjustments/remove_filters_adjustment.hpp"
#include "adjustments/remove_function_params_adjustment.hpp"

namespace ovms {
ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forMinja() {
    ChatTemplateAdjusterConfig config;
    config.removeFunctions = {"min", "from_json"};
    config.removeFunctionsParams = {"tojson"};
    return config;
}

ChatTemplateAdjusterConfig ChatTemplateAdjusterConfig::forJinja() {
    ChatTemplateAdjusterConfig config;
    return config;
}

ChatTemplateAdjuster::ChatTemplateAdjuster(const ChatTemplateAdjusterConfig& config) {
    if (!config.removeFunctionsParams.empty() || !config.removeFunctions.empty()) {
        std::vector<std::string> removeFunctionsParams = config.removeFunctionsParams;
        removeFunctionsParams.insert(removeFunctionsParams.end(), config.removeFunctions.begin(), config.removeFunctions.end());
        if (!removeFunctionsParams.empty()) {
            adjustments.emplace_back(std::make_unique<RemoveFunctionParamsAdjustment>(removeFunctionsParams));
        }
    }
    if (!config.removeFunctions.empty()) {
        adjustments.emplace_back(std::make_unique<RemoveFiltersAdjustment>(config.removeFunctions));
    }
}

void ChatTemplateAdjuster::adjust(std::string& templateSource) const {
    for (const auto& adjustment : adjustments) {
        adjustment->apply(templateSource);
    }
}

}  // namespace ovms
