//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include "../multi_part_parser.hpp"

#pragma warning(push)
#pragma warning(disable : 6326)
#include <drogon/drogon.h>
#pragma warning(pop)

#include <string>
#include <string_view>
#include <memory>

namespace ovms {

class DrogonMultiPartParser : public MultiPartParser {
    bool hasParseError_{true};
    const drogon::HttpRequestPtr request{nullptr};
    const std::shared_ptr<drogon::MultiPartParser> parser{nullptr};

public:
    DrogonMultiPartParser(const drogon::HttpRequestPtr& request) :
        request(request),
        parser(std::make_shared<drogon::MultiPartParser>()) {}

    bool parse() override;

    bool hasParseError() const override;

    std::string getFieldByName(const std::string& name) const override;
    std::string_view getFileContentByName(const std::string& name) const override;
};

}  // namespace ovms
