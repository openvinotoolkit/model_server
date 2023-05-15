//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <exception>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>

#include "../logging.hpp"
#include "../modelversionstatus.hpp"

namespace ovms {

enum class PipelineDefinitionStateCode {
    BEGIN,
    RELOADING,
    LOADING_PRECONDITION_FAILED,
    LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION,
    AVAILABLE_REQUIRED_REVALIDATION,
    AVAILABLE,
    RETIRED
};

const std::string& pipelineDefinitionStateCodeToString(PipelineDefinitionStateCode code);

template <typename... States>
class MachineState {
public:
    MachineState(const std::string& type, const std::string& name) :
        type(type),
        name(name) {}
    template <typename Event>
    void handle(const Event& event) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "{}: {} state: {} handling: {}: {}",
            type, name, pipelineDefinitionStateCodeToString(getStateCode()), event.name, event.getDetails());
        try {
            std::visit([this, &event](auto state) { state->handle(event).execute(*this); }, currentState);
        } catch (std::logic_error& le) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "{}: {} state: {} handling: {} error: {}",
                type, name, pipelineDefinitionStateCodeToString(getStateCode()), event.name, le.what());
            throw;
        }
        SPDLOG_LOGGER_INFO(modelmanager_logger, "{}: {} state changed to: {} after handling: {}: {}",
            type, name, pipelineDefinitionStateCodeToString(getStateCode()), event.name, event.getDetails());
    }

    template <typename State>
    void changeStateTo() {
        currentState = &std::get<State>(allPossibleStates);
    }
    PipelineDefinitionStateCode getStateCode() const {
        while (true) {
            try {
                return std::visit([](const auto state) { return state->getStateCode(); }, currentState);
            } catch (const std::bad_variant_access&) {
                continue;
            }
        }
    }

private:
    const std::string type;
    const std::string& name;
    std::tuple<States...> allPossibleStates;
    std::variant<States*...> currentState{&std::get<0>(allPossibleStates)};
};
/**
 * State in which pipeline is only defined
 */
struct BeginState;
/**
 * State in which pipeline is available
 */
struct AvailableState;
/**
 * State in which pipeline is available
 * but there is revalidation pending since we know that one of used
 * models changed
 */
struct AvailableRequiredRevalidation;
/**
 * State in which pipeline is reloading
 */
struct ReloadState;
/**
 * State in which pipeline is defined in config and failed validation.
 */
struct LoadingPreconditionFailedState;
/**
 * State in which pipeline is defined in config, failed validation,
 * but there is revalidation pending since we know that one of used
 * models changed
 */
struct LoadingFailedLastValidationRequiredRevalidation;
/**
 * State in which pipeline is retired - removed from config
 */
struct RetiredState;

#define EVENT_STRUCT_WITH_NAME(x)               \
    struct x {                                  \
        static constexpr const char* name = #x; \
        x(const std::string& details = "") :    \
            details(details) {}                 \
        const std::string& getDetails() const { \
            return details;                     \
        }                                       \
                                                \
    private:                                    \
        const std::string details;              \
    };

EVENT_STRUCT_WITH_NAME(ReloadEvent);
EVENT_STRUCT_WITH_NAME(ValidationFailedEvent);
EVENT_STRUCT_WITH_NAME(ValidationPassedEvent);
EVENT_STRUCT_WITH_NAME(UsedModelChangedEvent);
EVENT_STRUCT_WITH_NAME(RetireEvent);

template <typename State>
struct StateChanger {
    template <typename MachineState>
    void execute(MachineState& pds) {
        pds.template changeStateTo<State>();
    }
};

struct StateKeeper {
    template <typename MachineState>
    void execute(MachineState& machine) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Keeping state");
    }
};

struct BeginState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::BEGIN;
    PipelineDefinitionStateCode getStateCode() const;
    StateChanger<ReloadState> handle(const ReloadEvent& e) const;
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const;
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const;
    StateKeeper handle(const UsedModelChangedEvent& e) const;
    StateKeeper handle(const RetireEvent& e) const;
};

struct ReloadState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::RELOADING;
    PipelineDefinitionStateCode getStateCode() const;
    StateKeeper handle(const ReloadEvent& e) const;
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const;
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const;
    StateKeeper handle(const UsedModelChangedEvent& e) const;
    StateKeeper handle(const RetireEvent& e) const;
};

struct AvailableState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::AVAILABLE;
    PipelineDefinitionStateCode getStateCode() const;
    StateChanger<ReloadState> handle(const ReloadEvent& e) const;
    StateKeeper handle(const ValidationPassedEvent& e) const;
    StateKeeper handle(const ValidationFailedEvent& e) const;
    StateChanger<AvailableRequiredRevalidation> handle(const UsedModelChangedEvent& e) const;
    StateChanger<RetiredState> handle(const RetireEvent& e) const;
};

struct AvailableRequiredRevalidation {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION;
    PipelineDefinitionStateCode getStateCode() const;
    StateChanger<ReloadState> handle(const ReloadEvent& e) const;
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const;
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const;
    StateKeeper handle(const UsedModelChangedEvent& e) const;
    StateChanger<RetiredState> handle(const RetireEvent& e) const;
};

struct LoadingPreconditionFailedState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED;
    PipelineDefinitionStateCode getStateCode() const;
    StateChanger<ReloadState> handle(const ReloadEvent& e) const;
    StateKeeper handle(const ValidationPassedEvent& e) const;
    StateKeeper handle(const ValidationFailedEvent& e) const;
    StateChanger<LoadingFailedLastValidationRequiredRevalidation> handle(const UsedModelChangedEvent& e) const;
    StateChanger<RetiredState> handle(const RetireEvent& e) const;
};

struct LoadingFailedLastValidationRequiredRevalidation {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION;
    PipelineDefinitionStateCode getStateCode() const;
    StateChanger<ReloadState> handle(const ReloadEvent& e) const;
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const;
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const;
    StateKeeper handle(const UsedModelChangedEvent& e) const;
    StateChanger<RetiredState> handle(const RetireEvent& e) const;
};

struct RetiredState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::RETIRED;
    PipelineDefinitionStateCode getStateCode() const;
    StateChanger<ReloadState> handle(const ReloadEvent& e) const;
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const;
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const;
    StateKeeper handle(const UsedModelChangedEvent& e) const;
    StateKeeper handle(const RetireEvent& e) const;
};

class PipelineDefinitionStatus : public MachineState<BeginState, ReloadState, AvailableState, AvailableRequiredRevalidation, LoadingPreconditionFailedState, LoadingFailedLastValidationRequiredRevalidation, RetiredState> {
public:
    PipelineDefinitionStatus(const std::string& type, const std::string& name);
    bool isAvailable() const;
    bool canEndLoaded() const;
    bool isRevalidationRequired() const;
    std::tuple<ModelVersionState, ModelVersionStatusErrorCode> convertToModelStatus() const;
};
}  // namespace ovms
