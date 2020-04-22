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

#include <utility>

namespace ovms {

/**
 * @enum Status
 * @brief This enum contains status codes for ovms functions
 */
enum class Status {
    OK,                     /*!< Success */
    PATH_INVALID,           /*!< The provided path is invalid or doesn't exists */
    FILE_INVALID,           /*!< File not found or cannot open */
    NETWORK_NOT_LOADED,     
    JSON_INVALID,           /*!< The file is not valid json */
    MODELINSTANCE_NOT_FOUND
};

/**
 * @enum ValidationStatusCode
 * @brief This enum contains status codes for ovms request validation
 */
enum class ValidationStatusCode {
    OK,                     /*!< Success */
    MODEL_NAME_MISSING,     /*!< Model with requested name is not found */
    MODEL_VERSION_MISSING,  /*!< Model with requested version is not found */
    INCORRECT_BATCH_SIZE,   /*!< Input batch size other than required */
    INVALID_INPUT_ALIAS,    /*!< Invalid number of inputs or name mismatch */
    INVALID_SHAPE,          /*!< Invalid shape dimension number or dimension value */
    INVALID_PRECISION,      /*!< Invalid precision */
    INVALID_CONTENT_SIZE,   /*!< Invalid content size */
    DESERIALIZATION_ERROR,  /*!< Error occured during deserialization */
    INFERENCE_ERROR,        /*!< Error occured during inference */
};


class ValidationStatus {
public:
    static const std::string& getError(const ValidationStatusCode code) {
        static const std::map<ValidationStatusCode, std::string> errors = {
            { ValidationStatusCode::OK,                     ""                                          },
            { ValidationStatusCode::MODEL_NAME_MISSING,     "Model with requested name is not found"    },
            { ValidationStatusCode::MODEL_VERSION_MISSING,  "Model with requested version is not found" },
            { ValidationStatusCode::INCORRECT_BATCH_SIZE,   "Incorrect batch size"                      },
            { ValidationStatusCode::INVALID_INPUT_ALIAS,    "Unexpected input tensor alias"             },
            { ValidationStatusCode::INVALID_SHAPE,          "Invalid input shape"                       },
            { ValidationStatusCode::INVALID_PRECISION,      "Invalid input precision"                   },
            { ValidationStatusCode::INVALID_CONTENT_SIZE,   "Invalid content size"                      },
            { ValidationStatusCode::DESERIALIZATION_ERROR,  "Error occured during deserialization"      },
            { ValidationStatusCode::INFERENCE_ERROR,        "Error occured during inference"            },
        };

        return errors.find(code)->second;
    }
};

} // namespace ovms
