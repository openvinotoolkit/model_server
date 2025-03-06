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
#include "multi_part_parser_drogon_impl.hpp"

namespace ovms {

bool DrogonMultiPartParser::parse() {
    this->hasParseError_ = this->parser->parse(request) != 0;
    return !this->hasParseError_;
}

bool DrogonMultiPartParser::hasParseError() const {
    return this->hasParseError_;
}

std::string DrogonMultiPartParser::getFieldByName(const std::string& name) const {
    return this->parser->getParameter<std::string>(name);
}

std::string_view DrogonMultiPartParser::getFileContentByName(const std::string& name) const {
    auto it = this->parser->getFilesMap().find(name);
    if (it == this->parser->getFilesMap().end()) {
        return "";
    }
    return it->second.fileContent();
}

}  // namespace ovms
