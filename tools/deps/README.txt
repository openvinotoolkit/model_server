#
# Copyright (c) 2020 Intel Corporation
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

OpenVINO Model Server makefile creates those dependency archives. Via `tools_get_deps` target, using $GIT/tools/deps/ toolset.

It contains ./bin/ directory with RPM files installed during release image docker build process.
It has also a ./src/ directory, with (L)GPL sources we need to put in the image.
The ./internal_src/ are sources we don't need to distribute (saving disk space + download time), but we may need them later.
