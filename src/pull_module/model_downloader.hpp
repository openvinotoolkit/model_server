#pragma once
//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <string>

namespace ovms {
class Status;
class IModelDownloader {
public:
    virtual ~IModelDownloader() = default;
    virtual Status downloadModel() = 0;
    virtual std::string getGraphDirectory() = 0;
    static Status checkIfOverwriteAndRemove(const std::string& path, const bool overwriteFlag);
};

}  // namespace ovms
