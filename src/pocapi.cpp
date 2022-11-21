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
#include "pocapi.hpp"

#include <cstdint>
#include <string>

#include "poc_api_impl.hpp"

OVMS_Status* OVMS_ServerGeneralOptionsNew(OVMS_ServerGeneralOptions** options) {
    *options = (OVMS_ServerGeneralOptions*)new ovms::GeneralOptionsImpl;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsDelete(OVMS_ServerGeneralOptions* options) {
    delete (ovms::GeneralOptionsImpl*)options;
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsNew(OVMS_ServerMultiModelOptions** options) {
    *options = (OVMS_ServerMultiModelOptions*)new ovms::MultiModelOptionsImpl;
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsDelete(OVMS_ServerMultiModelOptions* options) {
    delete (ovms::MultiModelOptionsImpl*)options;
    return 0;
}

OVMS_Status* OVMS_ServerNew(OVMS_Server** server) {
    *server = (OVMS_Server*)new ovms::ServerImpl;
    return 0;
}

OVMS_Status* OVMS_ServerDelete(OVMS_Server* server) {
    delete (ovms::ServerImpl*)server;
    return 0;
}

OVMS_Status* OVMS_ServerStartFromConfigurationFile(OVMS_Server* server,
    OVMS_ServerGeneralOptions* general_options,
    OVMS_ServerMultiModelOptions* multi_model_specific_options) {
    ovms::ServerImpl* srv = (ovms::ServerImpl*)server;
    ovms::GeneralOptionsImpl* go = (ovms::GeneralOptionsImpl*)general_options;
    ovms::MultiModelOptionsImpl* mmo = (ovms::MultiModelOptionsImpl*)multi_model_specific_options;
    std::int64_t res = srv->start(go, mmo);
    return (OVMS_Status*)res;  // TODO: Return proper OVMS_Status instead of a raw status code
}

OVMS_Status* OVMS_ServerGeneralOptionsSetGrpcPort(OVMS_ServerGeneralOptions* options,
    uint64_t grpcPort) {
    ovms::GeneralOptionsImpl* go = (ovms::GeneralOptionsImpl*)options;
    go->grpcPort = grpcPort;
    return 0;
}

OVMS_Status* OVMS_ServerGeneralOptionsSetRestPort(OVMS_ServerGeneralOptions* options,
    uint64_t restPort) {
    ovms::GeneralOptionsImpl* go = (ovms::GeneralOptionsImpl*)options;
    go->restPort = restPort;
    return 0;
}

OVMS_Status* OVMS_ServerMultiModelOptionsSetConfigPath(OVMS_ServerMultiModelOptions* options,
    const char* config_path) {
    ovms::MultiModelOptionsImpl* mmo = (ovms::MultiModelOptionsImpl*)options;
    mmo->configPath = std::string(config_path);
    return 0;
}
