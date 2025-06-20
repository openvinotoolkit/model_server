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
#pragma once

#include <string>
#include <vector>

namespace ovms {

class MultiPartParser {
public:
    // Called by ovms core during initial request processing to deduce servable name for routing
    virtual bool parse() = 0;

    // API for MP calculators to check whether request was an actual multipart request
    virtual bool hasParseError() const = 0;

    // API for MP calculators to get the multipart field content by field name.
    // Returns empty string if field is not found.
    virtual std::string getFieldByName(const std::string& name) const = 0;

    // API for MP calculators to get the multipart file content by field name.
    // Returns empty string if file is not found.
    virtual std::string_view getFileContentByFieldName(const std::string& name) const = 0;

    virtual std::vector<std::string_view> getFilesArrayByFieldName(const std::string& name) const = 0;
};

}  // namespace ovms
