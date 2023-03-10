#
# Copyright (c) 2023 Intel Corporation
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
    name = "mediapipe_calculators",
    srcs = [],
    hdrs = [],
    visibility = ["//visibility:public"],
    deps = [
        "@google_mediapipe//mediapipe/calculators/openvino:openvino_converter_calculator",
        "@google_mediapipe//mediapipe/calculators/openvino:openvino_inference_calculator",
        "@google_mediapipe//mediapipe/calculators/openvino:openvino_tensors_to_classification_calculator",
        "@google_mediapipe//mediapipe/calculators/openvino:openvino_tensors_to_detections_calculator",
        "@google_mediapipe//mediapipe/calculators/core:concatenate_vector_calculator",
        "@google_mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "@google_mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        "@google_mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@google_mediapipe//mediapipe/calculators/core:add_header_calculator",
        "@google_mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@google_mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@google_mediapipe//mediapipe/calculators/core:concatenate_vector_calculator_hdr",
        "@google_mediapipe//mediapipe/calculators/core:concatenate_detection_vector_calculator",
        "@google_mediapipe//mediapipe/calculators/core:concatenate_proto_list_calculator",
        "@google_mediapipe//mediapipe/calculators/core:clip_vector_size_calculator",
        "@google_mediapipe//mediapipe/calculators/core:clip_detection_vector_size_calculator",
        "@google_mediapipe//mediapipe/calculators/core:counting_source_calculator",
        "@google_mediapipe//mediapipe/calculators/core:make_pair_calculator",
        "@google_mediapipe//mediapipe/calculators/core:matrix_multiply_calculator",
        "@google_mediapipe//mediapipe/calculators/core:matrix_subtract_calculator",
        "@google_mediapipe//mediapipe/calculators/core:mux_calculator",
        "@google_mediapipe//mediapipe/calculators/core:non_zero_calculator",
        "@google_mediapipe//mediapipe/calculators/core:packet_cloner_calculator",
        "@google_mediapipe//mediapipe/calculators/core:packet_inner_join_calculator",
        "@google_mediapipe//mediapipe/calculators/core:packet_thinner_calculator",
        "@google_mediapipe//mediapipe/calculators/core:pass_through_calculator",
        "@google_mediapipe//mediapipe/calculators/core:round_robin_demux_calculator",
        "@google_mediapipe//mediapipe/calculators/core:immediate_mux_calculator",
        "@google_mediapipe//mediapipe/calculators/core:packet_presence_calculator",
        "@google_mediapipe//mediapipe/calculators/core:string_to_int_calculator",
        "@google_mediapipe//mediapipe/calculators/core:default_side_packet_calculator",
        "@google_mediapipe//mediapipe/calculators/core:packet_resampler_calculator",
        "@google_mediapipe//mediapipe/calculators/core:split_proto_list_calculator",
        "@google_mediapipe//mediapipe/calculators/core:dequantize_byte_array_calculator",
        "@google_mediapipe//mediapipe/calculators/core:quantize_float_vector_calculator",
        "@google_mediapipe//mediapipe/calculators/core:sequence_shift_calculator",
        "@google_mediapipe//mediapipe/calculators/core:gate_calculator",
        "@google_mediapipe//mediapipe/calculators/core:matrix_to_vector_calculator",
        "@google_mediapipe//mediapipe/calculators/core:merge_calculator",
        "@google_mediapipe//mediapipe/calculators/core:stream_to_side_packet_calculator",
        "@google_mediapipe//mediapipe/calculators/core:constant_side_packet_calculator",
        "@google_mediapipe//mediapipe/calculators/core:graph_profile_calculator",
        "@google_mediapipe//mediapipe/calculators/core:get_vector_item_calculator",
        "@google_mediapipe//mediapipe/calculators/core:vector_size_calculator",
        "@google_mediapipe//mediapipe/calculators/core:packet_sequencer_calculator",
        "@google_mediapipe//mediapipe/calculators/core:merge_to_vector_calculator",
        "@google_mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@google_mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@google_mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "@google_mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@google_mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@google_mediapipe//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "@google_mediapipe//mediapipe/calculators/video:opencv_video_encoder_calculator",
    ],
    alwayslink = 1,
)