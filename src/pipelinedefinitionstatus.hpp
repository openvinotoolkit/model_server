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

#include <spdlog/spdlog.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#pragma GCC diagnostic pop

#include "status.hpp"

namespace ovms {

enum class PipelineDefinitionStateCode {
    BEGIN,
    LOADING_PRECONDITION_FAILED,
    LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION,
    AVAILABLE_REQUIRED_REVALIDATION,
    AVAILABLE,
    RETIRED
};

inline const std::string& pipelineDefinitionStateCodeToString(PipelineDefinitionStateCode code) {
    static const std::unordered_map<PipelineDefinitionStateCode, std::string> names{
        {PipelineDefinitionStateCode::BEGIN, "BEGIN"},
        {PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED, "LOADING_PRECONDITION_FAILED"},
        {PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION, "LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION"},
        {PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION, "AVAILABLE_REQUIRED_REVALIDATION"},
        {PipelineDefinitionStateCode::AVAILABLE, "AVAILABLE"},
        {PipelineDefinitionStateCode::RETIRED, "RETIRED"}};
    return names.at(code);
}

template <typename... States>
class MachineState {
public:
    template <typename Event>
    void handle(const Event& event) {
        std::visit([this, &event](auto state) { state->handle(event).execute(*this); }, currentState);
    }

    template <typename State>
    void changeStateTo() {
        currentState = &std::get<State>(allPossibleStates);
    }
    void printState() const {
        std::visit([](const auto state) { state->print(); }, currentState);
    }
    PipelineDefinitionStateCode getStateCode() const {
        return std::visit([](const auto state) { return state->getStateCode(); }, currentState);
    }

private:
    std::tuple<States...> allPossibleStates;
    std::variant<States*...> currentState{&std::get<0>(allPossibleStates)};
};

struct BeginState;
struct AvailableState;
struct AvailableRequiredRevalidation;
struct LoadingPreconditionFailedState;
struct LoadingFailedLastValidationRequiredRevalidation;
struct RetiredState;

#define EVENT_STRUCT_WITH_NAME(x)               \
    struct x {                                  \
        static constexpr const char* name = #x; \
    };

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
        SPDLOG_DEBUG("Keeping state");
    }
};

constexpr const char* INVALID_TRANSITION_MESSAGE = "Tried to conduct invalid transition.";

struct BeginState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::BEGIN;
    PipelineDefinitionStateCode getStateCode() const {
        return code;
    }
    void print() const {
        SPDLOG_ERROR(pipelineDefinitionStateCodeToString(getStateCode()));
    }
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateKeeper handle(const UsedModelChangedEvent& e) const {
        SPDLOG_ERROR("State:{} handling: {}. {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name, INVALID_TRANSITION_MESSAGE);
        throw std::logic_error(INVALID_TRANSITION_MESSAGE);
        return {};
    }
    StateKeeper handle(const RetireEvent& e) const {
        SPDLOG_ERROR("State:{} handling: {}. {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name, INVALID_TRANSITION_MESSAGE);
        throw std::logic_error(INVALID_TRANSITION_MESSAGE);
        return {};
    }
};

struct AvailableState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::AVAILABLE;
    PipelineDefinitionStateCode getStateCode() const {
        return code;
    }
    void print() const {
        SPDLOG_ERROR(pipelineDefinitionStateCodeToString(getStateCode()));
    }
    StateKeeper handle(const ValidationPassedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<AvailableRequiredRevalidation> handle(const UsedModelChangedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<RetiredState> handle(const RetireEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
};

struct AvailableRequiredRevalidation {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION;
    PipelineDefinitionStateCode getStateCode() const {
        return code;
    }
    void print() const {
        SPDLOG_ERROR(pipelineDefinitionStateCodeToString(getStateCode()));
    }
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateKeeper handle(const UsedModelChangedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<RetiredState> handle(const RetireEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
};

struct LoadingPreconditionFailedState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED;
    PipelineDefinitionStateCode getStateCode() const {
        return code;
    }
    void print() const {
        SPDLOG_ERROR(pipelineDefinitionStateCodeToString(getStateCode()));
    }
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateKeeper handle(const ValidationFailedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<LoadingFailedLastValidationRequiredRevalidation> handle(const UsedModelChangedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<RetiredState> handle(const RetireEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
};

struct LoadingFailedLastValidationRequiredRevalidation {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION;
    PipelineDefinitionStateCode getStateCode() const {
        return code;
    }
    void print() const {
        SPDLOG_ERROR(pipelineDefinitionStateCodeToString(getStateCode()));
    }
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateKeeper handle(const UsedModelChangedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<RetiredState> handle(const RetireEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
};

struct RetiredState {
    static const PipelineDefinitionStateCode code = PipelineDefinitionStateCode::RETIRED;
    PipelineDefinitionStateCode getStateCode() const {
        return code;
    }
    void print() const {
        SPDLOG_ERROR(pipelineDefinitionStateCodeToString(getStateCode()));
    }
    StateChanger<AvailableState> handle(const ValidationPassedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateChanger<LoadingPreconditionFailedState> handle(const ValidationFailedEvent& e) const {
        SPDLOG_INFO("State:{} handling: {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name);
        return {};
    }
    StateKeeper handle(const UsedModelChangedEvent& e) const {
        SPDLOG_ERROR("State:{} handling: {}. {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name, INVALID_TRANSITION_MESSAGE);
        throw std::logic_error(INVALID_TRANSITION_MESSAGE);
        return {};
    }
    StateKeeper handle(const RetireEvent& e) const {
        SPDLOG_ERROR("State:{} handling: {}. {}", pipelineDefinitionStateCodeToString(getStateCode()), e.name, INVALID_TRANSITION_MESSAGE);
        throw std::logic_error(INVALID_TRANSITION_MESSAGE);
        return {};
    }
};

class PipelineDefinitionStatus : public MachineState<BeginState, AvailableState, AvailableRequiredRevalidation, LoadingPreconditionFailedState, LoadingFailedLastValidationRequiredRevalidation, RetiredState> {
public:
    bool isAvailable() const {
        auto state = getStateCode();
        return (state == PipelineDefinitionStateCode::AVAILABLE) ||
               (state == PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
    }
    bool canEndLoaded() const {
        auto state = getStateCode();
        return isAvailable() ||
               (state == PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION) ||
               (state == PipelineDefinitionStateCode::BEGIN);
    }
};

}  // namespace ovms
