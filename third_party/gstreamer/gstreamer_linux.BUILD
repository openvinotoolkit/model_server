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

# GStreamer headers for Linux.
#
# GStreamer is built from source in the gstreamer-builder Docker stage
# and installed to /opt/gstreamer (prefix).  The glib headers are copied
# into the same prefix so the tree is self-contained.
#
# Layout expected under /opt/gstreamer:
#   include/gstreamer-1.0/      GStreamer core + plugins headers
#   include/glib-2.0/           GLib/GObject headers (copied from system)
#   lib/gstreamer-1.0/include/  per-plugin generated headers (if any)
#
# NOTE: This target provides HEADERS ONLY — no linkopts.
# src/mpi:gst_loader dlopens the shared libs at runtime so the OVMS
# binary starts without GStreamer present on end-user systems.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gstreamer_headers",
    hdrs = glob([
        "include/gstreamer-1.0/**/*.h",
        "include/glib-2.0/**/*.h",
        "lib/gstreamer-1.0/include/**/*.h",
    ], allow_empty = True),
    includes = [
        "include/gstreamer-1.0",
        "include/glib-2.0",
        "lib/gstreamer-1.0/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gstreamer",
    deps = [":gstreamer_headers"],
    visibility = ["//visibility:public"],
)
