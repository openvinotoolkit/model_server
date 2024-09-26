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
#include "modelversionstatus.hpp"

#include <iostream>
#include <string>
#include <unordered_map>

#include "logging.hpp"

// note: think about using https://github.com/Neargye/magic_enum when compatible compiler is supported.

namespace ovms {

// those values have to match tensorflow-serving state:
static const std::unordered_map<ModelVersionState, std::string> versionStatesStrings = {
    {ModelVersionState::START, "START"},
    {ModelVersionState::LOADING, "LOADING"},
    {ModelVersionState::AVAILABLE, "AVAILABLE"},
    {ModelVersionState::UNLOADING, "UNLOADING"},
    {ModelVersionState::END, "END"}};
const std::string& ModelVersionStateToString(ModelVersionState state) {
    return versionStatesStrings.at(state);
}

static const std::unordered_map<ModelVersionStatusErrorCode, std::string> versionsStatesErrors = {
    {ModelVersionStatusErrorCode::OK, "OK"},
    {ModelVersionStatusErrorCode::UNKNOWN, "UNKNOWN"},
    {ModelVersionStatusErrorCode::FAILED_PRECONDITION, "FAILED_PRECONDITION"}};

const std::string& ModelVersionStatusErrorCodeToString(ModelVersionStatusErrorCode code) {
    return versionsStatesErrors.at(code);
}

ModelVersionStatus::ModelVersionStatus(const std::string& model_name, model_version_t version, ModelVersionState state) :
    modelName(model_name),
    version(version),
    state(state),
    errorCode(ModelVersionStatusErrorCode::OK) {
    logStatus();
}

ModelVersionState ModelVersionStatus::getState() const {
    return this->state;
}

const std::string& ModelVersionStatus::getStateString() const {
    return ModelVersionStateToString(this->state);
}

ModelVersionStatusErrorCode ModelVersionStatus::getErrorCode() const {
    return this->errorCode;
}

const std::string& ModelVersionStatus::getErrorMsg() const {
    return ModelVersionStatusErrorCodeToString(this->errorCode);
}

bool ModelVersionStatus::willEndUnloaded() const {
    return ovms::ModelVersionState::UNLOADING <= this->state;
}

bool ModelVersionStatus::isFailedLoading() const {
    return this->state == ovms::ModelVersionState::LOADING && this->errorCode == ovms::ModelVersionStatusErrorCode::UNKNOWN;
}

void ModelVersionStatus::setLoading(ModelVersionStatusErrorCode error_code) {
    SPDLOG_DEBUG("{}: {} - {} (previous state: {}) -> error: {}", __func__, this->modelName, this->version, ModelVersionStateToString(this->state), ModelVersionStatusErrorCodeToString(error_code));
    state = ModelVersionState::LOADING;
    errorCode = error_code;
    logStatus();
}

void ModelVersionStatus::setAvailable(ModelVersionStatusErrorCode error_code) {
    SPDLOG_DEBUG("{}: {} - {} (previous state: {}) -> error: {}", __func__, this->modelName, this->version, ModelVersionStateToString(this->state), ModelVersionStatusErrorCodeToString(error_code));
    state = ModelVersionState::AVAILABLE;
    errorCode = error_code;
    logStatus();
}

void ModelVersionStatus::setUnloading(ModelVersionStatusErrorCode error_code) {
    SPDLOG_DEBUG("{}: {} - {} (previous state: {}) -> error: {}", __func__, this->modelName, this->version, ModelVersionStateToString(this->state), ModelVersionStatusErrorCodeToString(error_code));
    state = ModelVersionState::UNLOADING;
    errorCode = error_code;
    logStatus();
}

void ModelVersionStatus::setEnd(ModelVersionStatusErrorCode error_code) {
    SPDLOG_DEBUG("{}: {} - {} (previous state: {}) -> error: {}", __func__, this->modelName, this->version, ModelVersionStateToString(this->state), ModelVersionStatusErrorCodeToString(error_code));
    state = ModelVersionState::END;
    errorCode = error_code;
    logStatus();
}

void ModelVersionStatus::logStatus() {
    SPDLOG_INFO("STATUS CHANGE: Version {} of model {} status change. New status: ( \"state\": \"{}\", \"error_code\": \"{}\" )",
        this->version,
        this->modelName,
        ModelVersionStateToString(state),
        ModelVersionStatusErrorCodeToString(errorCode));
}

void ModelVersionStatus::setState(ModelVersionState state, ModelVersionStatusErrorCode error_code) {
    SPDLOG_DEBUG("{}: {} - {} (previous state: {}) -> error: {}", __func__, this->modelName, this->version, ModelVersionStateToString(this->state), ModelVersionStatusErrorCodeToString(error_code));
    this->state = state;
    errorCode = error_code;
    logStatus();
}
}  // namespace ovms
