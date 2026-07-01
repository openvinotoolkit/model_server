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
#include "config_status_utils.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "dags/pipelinedefinitionstatus.hpp"
#include "execution_context.hpp"
#include "metrics/metric.hpp"
#include "model.hpp"
#include "model_metric_reporter.hpp"
#include "modelinstance.hpp"
#include "model_instance_provider.hpp"
#include "modelversionstatus.hpp"
#include "servable_name_checker.hpp"
#include "single_version_servable_definition.hpp"
#include "status.hpp"

namespace ovms {

Status getAllModelsStatuses(ModelsStatuses& modelsStatuses, ModelInstanceProvider& modelProvider, ServableNameChecker& servableChecker, ExecutionContext context) {
    ModelsStatuses tmp;

    const auto servableNames = servableChecker.getServableDefinitionNames();
    for (const auto& servableName : servableNames) {
        std::vector<ModelVersionStatusDetails> versions;

        auto model_ptr = modelProvider.findModelByName(servableName);
        if (model_ptr) {
            auto modelVersionsInstances = model_ptr->getModelVersionsMapCopy();
            bool reported = false;
            for (const auto& [modelVersion, modelInstance] : modelVersionsInstances) {
                if (!reported) {
                    INCREMENT_IF_ENABLED(modelInstance.getMetricReporter().getGetModelStatusRequestSuccessMetric(context));
                    reported = true;
                }
                const auto& status = modelInstance.getStatus();
                ModelVersionStatusDetails details{
                    modelVersion,
                    status.getState(),
                    status.getErrorCode(),
                    status.getErrorMsg()};
                versions.push_back(std::move(details));
            }
        } else {
            auto* definition = servableChecker.findServableDefinition(servableName);
            if (!definition) {
                continue;
            }
            auto* svsd = dynamic_cast<SingleVersionServableDefinition*>(definition);
            if (!svsd) {
                continue;
            }
            INCREMENT_IF_ENABLED(svsd->getMetricReporter().getGetModelStatusRequestSuccessMetric(context));
            auto [state, error_code] = svsd->getStatus().convertToModelStatus();
            ModelVersionStatusDetails details{
                svsd->getVersion(),
                state,
                error_code,
                ModelVersionStatusErrorCodeToString(error_code)};
            versions.push_back(std::move(details));
        }

        tmp[servableName] = std::move(versions);
    }

    modelsStatuses.merge(tmp);
    return StatusCode::OK;
}

Status serializeModelsStatuses2Json(const ModelsStatuses& modelsStatuses, std::string& output) {
    if (modelsStatuses.empty()) {
        output = "{}";
        return StatusCode::OK;
    }

    std::string outputTmp;
    outputTmp += "{\n";

    bool firstModel = true;
    for (const auto& [modelName, versions] : modelsStatuses) {
        if (!firstModel) {
            outputTmp += ",\n";
        }
        firstModel = false;

        outputTmp += "\"" + modelName + "\" : \n{\n \"model_version_status\": [";

        if (versions.empty()) {
            outputTmp += "]\n}";
        } else {
            outputTmp += "\n";
            bool firstVersion = true;
            for (const auto& v : versions) {
                if (!firstVersion) {
                    outputTmp += ",\n";
                }
                firstVersion = false;

                outputTmp += "  {\n";
                outputTmp += "   \"version\": \"" + std::to_string(v.version) + "\",\n";
                outputTmp += "   \"state\": \"" + ModelVersionStateToString(v.state) + "\",\n";
                outputTmp += "   \"status\": {\n";
                outputTmp += "    \"error_code\": \"" + ModelVersionStatusErrorCodeToString(v.errorCode) + "\",\n";
                outputTmp += "    \"error_message\": \"" + v.errorMessage + "\"\n";
                outputTmp += "   }\n";
                outputTmp += "  }";
            }
            outputTmp += "\n ]\n}";
        }
    }

    outputTmp += "\n}";
    output = std::move(outputTmp);

    return StatusCode::OK;
}

}  // namespace ovms
