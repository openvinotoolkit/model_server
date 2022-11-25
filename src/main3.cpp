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
#include <iostream>

#include "pocapi.hpp"

int main(int argc, char** argv) {
    OVMS_ServerGeneralOptions* go = 0;
    OVMS_ServerMultiModelOptions* mmo = 0;
    OVMS_Server* srv;

    OVMS_ServerGeneralOptionsNew(&go);
    OVMS_ServerMultiModelOptionsNew(&mmo);
    OVMS_ServerNew(&srv);

    OVMS_ServerGeneralOptionsSetGrpcPort(go, 11337);
    OVMS_ServerGeneralOptionsSetRestPort(go, 11338);
    OVMS_ServerMultiModelOptionsSetConfigPath(mmo, "/ovms/src/test/c_api/config.json");

    OVMS_Status* res = OVMS_ServerStartFromConfigurationFile(srv, go, mmo);

    OVMS_ServerDelete(srv);
    OVMS_ServerMultiModelOptionsDelete(mmo);
    OVMS_ServerGeneralOptionsDelete(go);

    if (res == 0) {
        std::cout << "Finish with success" << std::endl;
    } else {
        std::cout << "Finish with fail" << std::endl;
    }

    return 0;
}
