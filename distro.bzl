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
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@mediapipe//mediapipe/framework:more_selects.bzl", "more_selects")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
def distro_flag():
    string_flag(
        name = "distro",
        values = ["redhat", "ubuntu"],
        build_setting_default = "ubuntu",
    )
    native.config_setting(
        name = "redhat_build",
        flag_values = {
            "distro": "redhat",
        },
    )
    native.config_setting(
        name = "ubuntu_build",
        flag_values = {
            "distro": "ubuntu",
        },
    )
    more_selects.config_setting_negation(
        name = "not_redhat_build",
        negate = ":redhat_build",
    )
    more_selects.config_setting_negation(
        name = "not_ubuntu_build",
        negate = ":ubuntu_build",
    )

# Controls whether espeak-ng is built from source (via Bazel) and bundled
# into the OVMS release. When "off", no espeak-ng artifacts are produced
# and the runtime will not have phonemization fallback available.
def espeak_flag():
    string_flag(
        name = "espeak",
        values = ["on", "off"],
        build_setting_default = "on",
    )
    native.config_setting(
        name = "espeak_on",
        flag_values = {
            "espeak": "on",
        },
    )
    native.config_setting(
        name = "espeak_off",
        flag_values = {
            "espeak": "off",
        },
    )
    more_selects.config_setting_negation(
        name = "not_espeak_on",
        negate = ":espeak_on",
    )
