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
#include <thread>

#include "pocapi.hpp"

int main(int argc, char** argv) {
    std::thread t([&argv, &argc]() {
        OVMS_Start(argc, argv);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // get model instance and have a lock on reload
    float a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11};
    float b[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    OVMS_Infer((char*)"dummy", a, b);
    for (int i = 0; i < 10; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;
    std::cout << __LINE__ << "FINISHED, press ctrl+c to stop " << std::endl;
    t.join();
    std::cout << __LINE__ << "FINISHED" << std::endl;
    return 0;
}
