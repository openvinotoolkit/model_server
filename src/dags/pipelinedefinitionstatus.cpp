//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include "pipelinedefinitionstatus.hpp"

#include <exception>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>

#include "../logging.hpp"
#include "../modelversionstatus.hpp"

namespace ovms {

const std::string& pipelineDefinitionStateCodeToString(PipelineDefinitionStateCode code) {
    static const std::unordered_map<PipelineDefinitionStateCode, std::string> names{
        {PipelineDefinitionStateCode::BEGIN, "BEGIN"},
        {PipelineDefinitionStateCode::RELOADING, "RELOADING"},
        {PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED, "LOADING_PRECONDITION_FAILED"},
        {PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION, "LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION"},
        {PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION, "AVAILABLE_REQUIRED_REVALIDATION"},
        {PipelineDefinitionStateCode::AVAILABLE, "AVAILABLE"},
        {PipelineDefinitionStateCode::RETIRED, "RETIRED"}};
    return names.at(code);
}

constexpr const char* INVALID_TRANSITION_MESSAGE = "Tried to conduct invalid transition.";

PipelineDefinitionStateCode BeginState::getStateCode() const {
    return code;
}
StateChanger<ReloadState> BeginState::handle(const ReloadEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateChanger<AvailableState> BeginState::handle(const ValidationPassedEvent& e) const {
    return {};
}
StateChanger<LoadingPreconditionFailedState> BeginState::handle(const ValidationFailedEvent& e) const {
    return {};
}
StateKeeper BeginState::handle(const UsedModelChangedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateKeeper BeginState::handle(const RetireEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}

PipelineDefinitionStateCode ReloadState::getStateCode() const {
    return code;
}
StateKeeper ReloadState::handle(const ReloadEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateChanger<AvailableState> ReloadState::handle(const ValidationPassedEvent& e) const {
    return {};
}
StateChanger<LoadingPreconditionFailedState> ReloadState::handle(const ValidationFailedEvent& e) const {
    return {};
}
StateKeeper ReloadState::handle(const UsedModelChangedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateKeeper ReloadState::handle(const RetireEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}

PipelineDefinitionStateCode AvailableState::getStateCode() const {
    return code;
}
StateChanger<ReloadState> AvailableState::handle(const ReloadEvent& e) const {
    return {};
}
StateKeeper AvailableState::handle(const ValidationPassedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateKeeper AvailableState::handle(const ValidationFailedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateChanger<AvailableRequiredRevalidation> AvailableState::handle(const UsedModelChangedEvent& e) const {
    return {};
}
StateChanger<RetiredState> AvailableState::handle(const RetireEvent& e) const {
    return {};
}

PipelineDefinitionStateCode AvailableRequiredRevalidation::getStateCode() const {
    return code;
}
StateChanger<ReloadState> AvailableRequiredRevalidation::handle(const ReloadEvent& e) const {
    return {};
}
StateChanger<AvailableState> AvailableRequiredRevalidation::handle(const ValidationPassedEvent& e) const {
    return {};
}
StateChanger<LoadingPreconditionFailedState> AvailableRequiredRevalidation::handle(const ValidationFailedEvent& e) const {
    return {};
}
StateKeeper AvailableRequiredRevalidation::handle(const UsedModelChangedEvent& e) const {
    return {};
}
StateChanger<RetiredState> AvailableRequiredRevalidation::handle(const RetireEvent& e) const {
    return {};
}

PipelineDefinitionStateCode LoadingPreconditionFailedState::getStateCode() const {
    return code;
}
StateChanger<ReloadState> LoadingPreconditionFailedState::handle(const ReloadEvent& e) const {
    return {};
}
StateKeeper LoadingPreconditionFailedState::handle(const ValidationPassedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateKeeper LoadingPreconditionFailedState::handle(const ValidationFailedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateChanger<LoadingFailedLastValidationRequiredRevalidation> LoadingPreconditionFailedState::handle(const UsedModelChangedEvent& e) const {
    return {};
}
StateChanger<RetiredState> LoadingPreconditionFailedState::handle(const RetireEvent& e) const {
    return {};
}

PipelineDefinitionStateCode LoadingFailedLastValidationRequiredRevalidation::getStateCode() const {
    return code;
}
StateChanger<ReloadState> LoadingFailedLastValidationRequiredRevalidation::handle(const ReloadEvent& e) const {
    return {};
}
StateChanger<AvailableState> LoadingFailedLastValidationRequiredRevalidation::handle(const ValidationPassedEvent& e) const {
    return {};
}
StateChanger<LoadingPreconditionFailedState> LoadingFailedLastValidationRequiredRevalidation::handle(const ValidationFailedEvent& e) const {
    return {};
}
StateKeeper LoadingFailedLastValidationRequiredRevalidation::handle(const UsedModelChangedEvent& e) const {
    return {};
}
StateChanger<RetiredState> LoadingFailedLastValidationRequiredRevalidation::handle(const RetireEvent& e) const {
    return {};
}

PipelineDefinitionStateCode RetiredState::getStateCode() const {
    return code;
}
StateChanger<ReloadState> RetiredState::handle(const ReloadEvent& e) const {
    return {};
}
StateChanger<AvailableState> RetiredState::handle(const ValidationPassedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateChanger<LoadingPreconditionFailedState> RetiredState::handle(const ValidationFailedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateKeeper RetiredState::handle(const UsedModelChangedEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}
StateKeeper RetiredState::handle(const RetireEvent& e) const {
    throw std::logic_error(INVALID_TRANSITION_MESSAGE);
    return {};
}

PipelineDefinitionStatus::PipelineDefinitionStatus(const std::string& type, const std::string& name) :
    MachineState(type, name) {}
bool PipelineDefinitionStatus::isAvailable() const {
    auto state = getStateCode();
    return (state == PipelineDefinitionStateCode::AVAILABLE) ||
           (state == PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
}
bool PipelineDefinitionStatus::canEndLoaded() const {
    auto state = getStateCode();
    return isAvailable() ||
           (state == PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION) ||
           (state == PipelineDefinitionStateCode::BEGIN) ||
           (state == PipelineDefinitionStateCode::RELOADING);
}
bool PipelineDefinitionStatus::isRevalidationRequired() const {
    auto state = getStateCode();
    return (state == PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION) ||
           (state == PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION);
}

std::tuple<ModelVersionState, ModelVersionStatusErrorCode> PipelineDefinitionStatus::convertToModelStatus() const {
    switch (getStateCode()) {
    case PipelineDefinitionStateCode::BEGIN:
    case PipelineDefinitionStateCode::RELOADING:
    case PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED_REQUIRED_REVALIDATION:
        return {
            ModelVersionState::LOADING,
            ModelVersionStatusErrorCode::OK};

    case PipelineDefinitionStateCode::LOADING_PRECONDITION_FAILED:
        return {
            ModelVersionState::LOADING,
            ModelVersionStatusErrorCode::FAILED_PRECONDITION};

    case PipelineDefinitionStateCode::AVAILABLE_REQUIRED_REVALIDATION:
    case PipelineDefinitionStateCode::AVAILABLE:
        return {
            ModelVersionState::AVAILABLE,
            ModelVersionStatusErrorCode::OK};

    case PipelineDefinitionStateCode::RETIRED:
        return {
            ModelVersionState::END,
            ModelVersionStatusErrorCode::OK};

    default:
        return {};
    }
}
}  // namespace ovms
