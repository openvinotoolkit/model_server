#
# Copyright (c) 2025 Intel Corporation
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

# GStreamer pre-built import for Windows (msvc_x86_64 installer layout)
#
# Expected install location: C:\Program Files\gstreamer\1.0\msvc_x86_64
# The new_local_repository in WORKSPACE points "path" there;
# this BUILD file maps headers and import libs for Bazel consumption.

package(default_visibility = ["//visibility:public"])

# ---------- headers ----------

cc_library(
    name = "gstreamer_headers",
    hdrs = glob([
        "include/gstreamer-1.0/**/*.h",
        "lib/gstreamer-1.0/include/**/*.h",
    ]),
    includes = [
        "include/gstreamer-1.0",
        "lib/gstreamer-1.0/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "glib_headers",
    hdrs = glob([
        "include/glib-2.0/**/*.h",
        "lib/glib-2.0/include/**/*.h",
    ]),
    includes = [
        "include/glib-2.0",
        "lib/glib-2.0/include",
    ],
    visibility = ["//visibility:public"],
)

# ---------- import libraries ----------

cc_import(
    name = "gstreamer_lib",
    interface_library = "lib/gstreamer-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "gstapp_lib",
    interface_library = "lib/gstapp-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "gstvideo_lib",
    interface_library = "lib/gstvideo-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "gstd3d11_lib",
    interface_library = "lib/gstd3d11-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "gstpbutils_lib",
    interface_library = "lib/gstpbutils-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "gstbase_lib",
    interface_library = "lib/gstbase-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "gstaudio_lib",
    interface_library = "lib/gstaudio-1.0.lib",
    system_provided = True,
)

cc_import(
    name = "glib_lib",
    interface_library = "lib/glib-2.0.lib",
    system_provided = True,
)

cc_import(
    name = "gobject_lib",
    interface_library = "lib/gobject-2.0.lib",
    system_provided = True,
)

# ---------- aggregate target ----------

cc_library(
    name = "gstreamer",
    deps = [
        ":gstreamer_headers",
        ":glib_headers",
        ":gstreamer_lib",
        ":gstapp_lib",
        ":gstvideo_lib",
        ":gstd3d11_lib",
        ":gstpbutils_lib",
        ":gstbase_lib",
        ":gstaudio_lib",
        ":glib_lib",
        ":gobject_lib",
    ],
    linkopts = [
        "d3d11.lib",
        "dxgi.lib",
    ],
    visibility = ["//visibility:public"],
)
