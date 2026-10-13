//*****************************************************************************
// Copyright 2026 Intel Corporation
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

// Test double for optimum-cli. Keep output deterministic and avoid invoking
// Python, downloading a model, or running a real conversion.
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: optimum-cli <command>" << std::endl;
        return 1;
    }

    std::cout << "Number of arguments: " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "Argument " << i << ": " << argv[i] << std::endl;
    }

    if (argv[1][0] == '-') {
        std::cout << "usage: optimum-cli" << std::endl;
    } else if (std::string(argv[1]) == "export") {
        std::cout << "Applying Weight Compression" << std::endl;
    }

    return 0;
}
