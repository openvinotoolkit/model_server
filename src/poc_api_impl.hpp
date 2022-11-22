#pragma once
//*****************************************************************************
// Copyright 2022 Intel Corporation
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
#include <cstdint>
#include <optional>
#include <string>

namespace ovms {

struct GeneralOptionsImpl {
    uint64_t grpcPort = 9178;
    uint64_t restPort = 0;
    std::string grpcBindAddress = "0.0.0.0";
    std::string restBindAddress = "0.0.0.0";
    uint grpcWorkers = 1;
    std::optional<uint> restWorkers;
    bool metricsEnabled = false;
    std::string metricsList;
    std::string cpuExtensionLibraryPath;
    std::string logLevel = "INFO";
    std::string logPath;
#ifdef MTR_ENABLED
    std::string tracePath;
#endif
    std::string grpcChannelArguments;
    uint filesystemPollWaitSeconds = 1;
    uint32_t sequenceCleanerPollWaitMinutes = 5;
    uint32_t resourcesCleanerPollWaitSeconds = 1;
    std::string cacheDir;
};

struct MultiModelOptionsImpl {
    std::string modelName;
    std::string modelPath;
    std::string batchSize;
    std::string shape;
    std::string layout;
    std::string modelVersionPolicy;
    uint32_t nireq = 0;
    std::string targetDevice;
    std::string pluginConfig;
    std::optional<bool> stateful;
    std::optional<bool> lowLatencyTransformation;
    std::optional<uint32_t> maxSequenceNumber;
    std::optional<bool> idleSequenceCleanup;

    std::string configPath;
};

struct ServerImpl {
    int start(GeneralOptionsImpl*, MultiModelOptionsImpl*);
};

}  // namespace ovms
