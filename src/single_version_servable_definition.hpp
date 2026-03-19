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
#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>

#include "modelversion.hpp"
#include "servable.hpp"
#include "servable_definition.hpp"
#include "status.hpp"
#include "tensorinfo_fwd.hpp"

namespace ovms {

class PipelineDefinitionStatus;
class ServableDefinitionUnloadGuard;
class StatusMetricReporter;
enum class PipelineDefinitionStateCode;

class SingleVersionServableDefinition : public ServableDefinition, public Servable {
    friend class ServableDefinitionUnloadGuard;

public:
    static constexpr model_version_t VERSION = 1;
    static constexpr uint64_t WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS = 500000;

    SingleVersionServableDefinition(const std::string& name) :
        Servable(name, VERSION) {}

    const std::string& getName() const override { return Servable::getName(); }
    model_version_t getVersion() const override { return Servable::getVersion(); }

    virtual const PipelineDefinitionStatus& getStatus() const = 0;
    virtual const tensor_map_t getInputsInfo() const = 0;
    virtual const tensor_map_t getOutputsInfo() const = 0;
    virtual StatusMetricReporter& getMetricReporter() const = 0;
    Status waitForLoaded(std::unique_ptr<ServableDefinitionUnloadGuard>& guard,
        uint32_t waitForLoadedTimeoutMicroseconds = WAIT_FOR_LOADED_DEFAULT_TIMEOUT_MICROSECONDS);

protected:
    std::atomic<uint64_t> requestsHandlesCounter = 0;
    std::condition_variable loadedNotify;

    void increaseRequestsHandlesCount() { ++requestsHandlesCounter; }
    void decreaseRequestsHandlesCount() { --requestsHandlesCounter; }

    virtual StatusCode notLoadedYetCode() const = 0;
    virtual StatusCode notLoadedAnymoreCode() const = 0;
};

}  // namespace ovms
