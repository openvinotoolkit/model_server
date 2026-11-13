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
cc_library(
    name = "decoder",
    srcs = [
        "codec/common/src/WelsTaskThread.cpp",
        "codec/common/src/WelsThread.cpp",
        "codec/common/src/WelsThreadLib.cpp",
        "codec/common/src/WelsThreadPool.cpp",
        "codec/common/src/common_tables.cpp",
        "codec/common/src/copy_mb.cpp",
        "codec/common/src/cpu.cpp",
        "codec/common/src/crt_util_safe_x.cpp",
        "codec/common/src/deblocking_common.cpp",
        "codec/common/src/expand_pic.cpp",
        "codec/common/src/intra_pred_common.cpp",
        "codec/common/src/mc.cpp",
        "codec/common/src/memory_align.cpp",
        "codec/common/src/sad_common.cpp",
        "codec/common/src/utils.cpp",
        "codec/common/src/welsCodecTrace.cpp",
        "codec/decoder/core/src/au_parser.cpp",
        "codec/decoder/core/src/bit_stream.cpp",
        "codec/decoder/core/src/cabac_decoder.cpp",
        "codec/decoder/core/src/deblocking.cpp",
        "codec/decoder/core/src/decode_mb_aux.cpp",
        "codec/decoder/core/src/decode_slice.cpp",
        "codec/decoder/core/src/decoder.cpp",
        "codec/decoder/core/src/decoder_core.cpp",
        "codec/decoder/core/src/decoder_data_tables.cpp",
        "codec/decoder/core/src/error_concealment.cpp",
        "codec/decoder/core/src/fmo.cpp",
        "codec/decoder/core/src/get_intra_predictor.cpp",
        "codec/decoder/core/src/manage_dec_ref.cpp",
        "codec/decoder/core/src/memmgr_nal_unit.cpp",
        "codec/decoder/core/src/mv_pred.cpp",
        "codec/decoder/core/src/parse_mb_syn_cabac.cpp",
        "codec/decoder/core/src/parse_mb_syn_cavlc.cpp",
        "codec/decoder/core/src/pic_queue.cpp",
        "codec/decoder/core/src/rec_mb.cpp",
        "codec/decoder/core/src/wels_decoder_thread.cpp",
        "codec/decoder/plus/src/welsDecoderExt.cpp",
    ],
    hdrs = glob([
        "codec/api/wels/*.h",
        "codec/common/inc/*.h",
        "codec/decoder/core/inc/*.h",
        "codec/decoder/plus/inc/*.h",
    ]),
    includes = [
        "codec/api/wels",
        "codec/common/inc",
        "codec/decoder/core/inc",
        "codec/decoder/plus/inc",
    ],
    linkopts = select({
        "@ovms//src:windows": [],
        "//conditions:default": ["-lpthread"],
    }),
    visibility = ["//visibility:public"],
)