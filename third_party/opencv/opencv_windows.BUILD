# Description:
#   OpenCV libraries for video/image processing on Windows
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

OPENCV_VERSION = "4100"  # 4.10.0

config_setting(
    name = "opt_build",
    values = {"compilation_mode": "opt"},
)

config_setting(
    name = "dbg_build",
    values = {"compilation_mode": "dbg"},
)

# The following build rule assumes that the executable "opencv-4.10.0-windows.exe"
# is downloaded and the files are extracted to local.
# If you install OpenCV separately, please modify the build rule accordingly.
cc_library(
    name = "opencv",
    srcs = select({
        ":opt_build": [
            "opencv_world" + OPENCV_VERSION + ".lib",
            "x64/vc16/bin/opencv_world" + OPENCV_VERSION + ".dll",
        ],
        ":dbg_build": [
            "opencv_world" + OPENCV_VERSION + "d.lib",
            "x64/vc16/bin/opencv_world" + OPENCV_VERSION + "d.dll",
        ],
    }),
    hdrs = glob(["include/opencv2/**/*.h*"]),
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
