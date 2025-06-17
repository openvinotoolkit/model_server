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
#include "cmd_exec.hpp"

#include <iostream>
#include <cstdio>
#include <memory>
#include <string>

namespace ovms {
std::string exec_cmd(const std::string& command, int returnCode) {
    char buffer[200];
    std::string result = "";
    try {
        // Open pipe to file
#ifdef _WIN32
        auto pcloseDeleter = [&returnCode](FILE* ptr) {
            if (ptr) {
                returnCode = _pclose(ptr);
            }
        };
        std::shared_ptr<FILE> pipe(_popen(command.c_str(), "r"), pcloseDeleter);
#elif __linux__
        auto pcloseDeleter = [&returnCode](FILE* ptr) {
            if (ptr) {
                returnCode = pclose(ptr);
            }
        };
        std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pcloseDeleter);
#endif
        if (!pipe) {
            return "Error: popen failed.";
        }

        // Read until end of process:
        while (fgets(buffer, sizeof(buffer), pipe.get()) != NULL) {
            result += buffer;
        }

        std::cout << "Command return code: " << returnCode << std::endl;
    } catch (const std::exception& e) {
        return std::string("Error occurred when running command: ") + e.what();
    } catch (...) {
        return "Error occurred when running command: ";
    }

    return result;
}

}  // namespace ovms
