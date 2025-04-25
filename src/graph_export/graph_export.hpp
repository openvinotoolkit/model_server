//****************************************************************************
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
#pragma once
#include <string>

namespace ovms {

class GraphExport {
protected:
    const std::string pipelineType;
    const std::string modelPath;
    const std::string maxNumSeqs;
    const std::string targetDevice;
    const std::string pluginConfig;
    const std::string enablePrefixCaching;
    const std::string cacheSize;
    const std::string maxNumBatchedTokens;
    bool dynamicSplitFuse;
    const std::string draftModelDirName;

    std::string graphString;

public:
    GraphExport(const std::string& pipelineType, const std::string& modelPath, const std::string& maxNumSeqs, const std::string& targetDevice,
        const std::string& pluginConfig, const std::string& enablePrefixCaching, const std::string& cacheSize, const std::string& maxNumBatchedTokens, bool dynamicSplitFuse,
        const std::string& draftModelDirName);

    createGraphFile(const std::string directoryPath);
};
}  // namespace ovms
