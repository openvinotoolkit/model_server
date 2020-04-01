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
};

} // namespace ovms
