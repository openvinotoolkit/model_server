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
    name = "streaming_audio_model",
    hdrs = ["streaming_audio_model.hpp"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "audio_utterance_buffer",
    srcs = ["audio_utterance_buffer.cpp"],
    hdrs = ["audio_utterance_buffer.hpp"],
    deps = [":streaming_audio_model"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "stt_audio_adapter",
    srcs = ["stt_audio_adapter.cpp"],
    hdrs = ["stt_audio_adapter.hpp"],
    deps = [
        ":streaming_audio_model",
        "//src:libovmslogging",
        "//third_party:genai",
        "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "voxtral_audio_adapter",
    srcs = ["voxtral_audio_adapter.cpp"],
    hdrs = ["voxtral_audio_adapter.hpp"],
    deps = [
        ":streaming_audio_model",
        ":stt_audio_adapter",
        "//third_party:genai",
        "//third_party:openvino",
    ],
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
    name = "webrtc_peer_connection",
    srcs = ["webrtc_peer_connection.cpp"],
    hdrs = [
        "rtp_packetization_config_utils.hpp",
        "webrtc_peer_connection.hpp",
    ],
    deps = [
        "//src:libovmslogging",
        ":opus_audio_codec",
        "@libdatachannel//:libdatachannel",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "webrtc_session_controller",
    srcs = ["webrtc_session_controller.cpp"],
    hdrs = ["webrtc_session_controller.hpp"],
    deps = [
        ":audio_utterance_buffer",
        ":opus_audio_codec",
        ":stt_audio_adapter",
        ":voxtral_audio_adapter",
        ":webrtc_peer_connection",
        "//src/audio/speech_to_text:stt_pipeline_handle",
        "//src/port:rapidjson_document",
        "//src/port:rapidjson_istreamwrapper",
        "//src/port:rapidjson_stringbuffer",
        "//src/port:rapidjson_writer",
        "//third_party:genai",
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
        ":opus_audio_codec",
        ":webrtc_peer_connection",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "webrtc_session_controller_test",
    srcs = ["webrtc_session_controller_test.cpp"],
    linkstatic = 1,
    deps = [
        "//src/audio:audio_utils",
        "//src/audio/speech_to_text:stt_pipeline_handle",
        "//third_party:genai",
        ":webrtc_session_controller",
        ":voxtral_audio_adapter",
        ":opus_audio_codec",
        "@libdatachannel//:libdatachannel",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "audio_utterance_buffer_test",
    srcs = ["audio_utterance_buffer_test.cpp"],
    linkstatic = 1,
    deps = [
        ":audio_utterance_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "opus_audio_codec_test",
    srcs = ["opus_audio_codec_test.cpp"],
    linkstatic = 1,
    deps = [
        ":opus_audio_codec",
        "@com_google_googletest//:gtest_main",
    ],
)
