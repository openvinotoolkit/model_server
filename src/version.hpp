//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#ifndef SRC_VERSION_HPP_
#define SRC_VERSION_HPP_
#define PROJECT_NAME "REPLACE_PROJECT_NAME"
#define PROJECT_VER_PATCH "REPLACE_PROJECT_PATCH"
#define OPENVINO_NAME "REPLACE_OPENVINO_NAME"
#endif  // SRC_VERSION_HPP_"

#include <string>

//http://repository.toolbox.iotg.sclab.intel.com/ov-packages/l_openvino_toolkit_p_2021.1.105.tgz
std::string GetOpenVinoVersionFromPackageUrl()
{
    std::string version_keyword = "l_openvino_toolkit_p_";
    std::string extension_keyword = ".tgz";
    std::string prefix = "OpenVINO backend ";
    std::string input_name = std::string(OPENVINO_NAME);
    int ver_start = input_name.find(version_keyword);
    int ver_end = input_name.find(extension_keyword);
    if (ver_start == std::string::npos || ver_end == std::string::npos || ver_end <= ver_start) {
        std::cout << "Warning:unsupported OpenVINO version string:" << OPENVINO_NAME << std::endl;
        return prefix + "unknown";
    }

    std::string ov_version = input_name.substr(ver_start + version_keyword.size() , input_name.size() - ver_start - version_keyword.size() - extension_keyword.size());
    return prefix + ov_version;
}
