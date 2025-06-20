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

std::string_view DrogonMultiPartParser::getFileContentByFieldName(const std::string& name) const {
    
    auto fileMap = this->parser->getFilesMap();
    // std::cout << "There are " << fileMap.size() << " files in the multipart request." << std::endl;

    // for (const auto& file : fileMap) {
    //     std::cout << "File name: " << file.first << ", size: " << file.second.fileLength() << " bytes." << std::endl;
    // }

    auto v = this->parser->getFiles();
    std::cout << "There are " << v.size() << " files in the multipart request." << std::endl;
    for (const auto& file : v) {
        std::cout << "File name: " << file.getFileName() << ", itemname: " << file.getItemName() << ", size: " << file.fileLength() << " bytes." << std::endl;
    }

    auto it = fileMap.find(name);
    if (it == fileMap.end()) {
        return "";
    }
    return it->second.fileContent();
}

std::vector<std::string_view> DrogonMultiPartParser::getFilesArrayByFieldName(const std::string& name) const {
    auto files = this->parser->getFiles();
    std::vector<std::string_view> result;
    for (const auto& file : files) {
        if (file.getItemName() == name) {  // take, it contains []
            result.push_back(file.fileContent());
        }
    }
    return result;
}

}  // namespace ovms
