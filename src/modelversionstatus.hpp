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

#include <iostream>
#include <string>
#include <unordered_map>

#include "modelversion.hpp"

// note: think about using https://github.com/Neargye/magic_enum when compatible compiler is supported.

namespace ovms {

// those values have to match tensorflow-serving state:
enum class ModelVersionState : int {
    START = 10,
    LOADING = 20,
    AVAILABLE = 30,
    UNLOADING = 40,
    END = 50
};

const std::string& ModelVersionStateToString(ModelVersionState state);

enum class ModelVersionStatusErrorCode : int {
    OK = 0,
    // CANCELLED = 1,
    UNKNOWN = 2,
    // INVALID_ARGUMENT = 3,
    // DEADLINE_EXCEEDED = 4,
    // NOT_FOUND = 5,
    // ALREADY_EXISTS = 6,
    // PERMISSION_DENIED = 7,
    // UNAUTHENTICATED = 16,
    // RESOURCE_EXHAUSTED = 8,
    FAILED_PRECONDITION = 9,
    // ABORTED = 10,
    // OUT_OF_RANGE = 11,
    // UNIMPLEMENTED = 12,
    // INTERNAL = 13,
    // UNAVAILABLE = 14,
    // DATA_LOSS = 15,
    // DO_NOT_USE_RESERVED_FOR_FUTURE_EXPANSION_USE_DEFAULT_IN_SWITCH_INSTEAD_
    //    = 20
};

const std::string& ModelVersionStatusErrorCodeToString(ModelVersionStatusErrorCode code);

class ModelVersionStatus {
    std::string modelName;
    model_version_t version;
    ModelVersionState state;
    ModelVersionStatusErrorCode errorCode;

public:
    ModelVersionStatus() = delete;

    ModelVersionStatus(const std::string& model_name, model_version_t version, ModelVersionState state = ModelVersionState::START);

    ModelVersionState getState() const;

    const std::string& getStateString() const;

    ModelVersionStatusErrorCode getErrorCode() const;

    const std::string& getErrorMsg() const;

    /**
     * @brief Check if current state is state that is either transforming to END or already in that state.
     *
     * @return
     */
    bool willEndUnloaded() const;

    bool isFailedLoading() const;
    void setLoading(ModelVersionStatusErrorCode error_code = ModelVersionStatusErrorCode::OK);

    void setAvailable(ModelVersionStatusErrorCode error_code = ModelVersionStatusErrorCode::OK);

    void setUnloading(ModelVersionStatusErrorCode error_code = ModelVersionStatusErrorCode::OK);

    void setEnd(ModelVersionStatusErrorCode error_code = ModelVersionStatusErrorCode::OK);
    void setState(ModelVersionState state, ModelVersionStatusErrorCode error_code = ModelVersionStatusErrorCode::OK);

private:
    void logStatus();
};

}  // namespace ovms
