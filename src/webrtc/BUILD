#
# Copyright (c) 2026 Intel Corporation
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

load("//:common_settings.bzl", "ovms_cc_library")

ovms_cc_library(
    name = "webrtc_signaling_session",
    srcs = ["webrtc_signaling_session.cpp"],
    hdrs = ["webrtc_signaling_session.hpp"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "streaming_audio_model",
    hdrs = ["streaming_audio_model.hpp"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "mock_echo_streaming_audio_model",
    srcs = ["mock_echo_streaming_audio_model.cpp"],
    hdrs = ["mock_echo_streaming_audio_model.hpp"],
    deps = [":streaming_audio_model"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "opus_audio_codec",
    srcs = ["opus_audio_codec.cpp"],
    hdrs = ["opus_audio_codec.hpp"],
    deps = ["@libopus//:libopus"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "streaming_audio_processor",
    srcs = ["streaming_audio_processor.cpp"],
    hdrs = ["streaming_audio_processor.hpp"],
    deps = [
        ":opus_audio_codec",
        ":streaming_audio_model",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "webrtc_peer_connection",
    srcs = ["webrtc_peer_connection.cpp"],
    hdrs = ["webrtc_peer_connection.hpp"],
    deps = [
        "//src:libovmslogging",
        ":streaming_audio_processor",
        "@libdatachannel//:libdatachannel",
    ],
    visibility = ["//visibility:public"],
)

# Standalone smoke test, isolated from the monolithic //src:ovms_test binary
# so this POC module can be iterated on without rebuilding the full test suite.
cc_test(
    name = "webrtc_peer_connection_test",
    srcs = ["webrtc_peer_connection_test.cpp"],
    # ovms_cc_library builds with -fvisibility=hidden, so cc_test's default
    # dynamic linking against the .so can't see these symbols; link statically.
    linkstatic = 1,
    deps = [
        ":mock_echo_streaming_audio_model",
        ":opus_audio_codec",
        ":streaming_audio_processor",
        ":webrtc_peer_connection",
        "@com_google_googletest//:gtest_main",
    ],
)


cc_test(
    name = "webrtc_signaling_session_test",
    srcs = ["webrtc_signaling_session_test.cpp"],
    linkstatic = 1,
    deps = [
        ":webrtc_signaling_session",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "streaming_audio_model_test",
    srcs = ["streaming_audio_model_test.cpp"],
    linkstatic = 1,
    deps = [
        ":mock_echo_streaming_audio_model",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "opus_audio_codec_test",
    srcs = ["opus_audio_codec_test.cpp"],
    linkstatic = 1,
    deps = [
        ":mock_echo_streaming_audio_model",
        ":opus_audio_codec",
        ":streaming_audio_processor",
        "@com_google_googletest//:gtest_main",
    ],
)

