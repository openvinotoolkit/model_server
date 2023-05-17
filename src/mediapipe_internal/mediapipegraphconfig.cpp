//*****************************************************************************
// Copyright 2023 Intel Corporation
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
#include "mediapipegraphconfig.hpp"

#include "../filesystem.hpp"

namespace ovms {
void MediapipeGraphConfig::setBasePath(const std::string& basePath) {
    FileSystem::setPath(this->basePath, basePath, this->rootDirectoryPath);
}

void MediapipeGraphConfig::setGraphPath(const std::string& graphPath) {
    FileSystem::setPath(this->graphPath, graphPath, this->basePath);
}

void MediapipeGraphConfig::setSubconfigPath(const std::string& subconfigPath) {
    FileSystem::setPath(this->subconfigPath, subconfigPath, this->basePath);
}

}  // namespace ovms
