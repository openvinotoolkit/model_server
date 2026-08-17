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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "azure_core",
    srcs = glob(["lib/libazure-core.so"]),
    hdrs = glob([
        "include/azure/core/**/*.hpp",
        "include/azure/core/**/*.h",
        "include/azure/core.hpp",
    ]),
    strip_include_prefix = "include",
    linkstatic = False,
    data = glob(["lib/libazure-core.so.*"]),
)

cc_library(
    name = "azure_storage_common",
    srcs = glob(["lib/libazure-storage-common.so"]),
    hdrs = glob([
        "include/azure/storage/common/**/*.hpp",
        "include/azure/storage/common/**/*.h",
    ]),
    strip_include_prefix = "include",
    linkstatic = False,
    data = glob(["lib/libazure-storage-common.so.*"]),
    deps = [":azure_core"],
)

cc_library(
    name = "azure_storage_blobs",
    srcs = glob(["lib/libazure-storage-blobs.so"]),
    hdrs = glob([
        "include/azure/storage/blobs/**/*.hpp",
        "include/azure/storage/blobs/**/*.h",
        "include/azure/storage/blobs.hpp",
    ]),
    strip_include_prefix = "include",
    linkstatic = False,
    data = glob(["lib/libazure-storage-blobs.so.*"]),
    deps = [":azure_storage_common"],
)

cc_library(
    name = "azure_storage_files_shares",
    srcs = glob(["lib/libazure-storage-files-shares.so"]),
    hdrs = glob([
        "include/azure/storage/files/**/*.hpp",
        "include/azure/storage/files/**/*.h",
        "include/azure/storage/files/shares.hpp",
    ]),
    strip_include_prefix = "include",
    linkstatic = False,
    data = glob(["lib/libazure-storage-files-shares.so.*"]),
    deps = [":azure_storage_common"],
)

cc_library(
    name = "storage",
    deps = [
        ":azure_storage_blobs",
        ":azure_storage_files_shares",
    ],
)
