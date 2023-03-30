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
    #TODO:ENABLE_GPU_WIP:defines = ["MEDIAPIPE_DISABLE_GPU=1"],
    deps = [
        #ENABLE_WTIH_OV_CALCULATORS"@mediapipe//mediapipe/calculators/openvino:openvino_converter_calculator",
        #ENABLE_WTIH_OV_CALCULATORS"@mediapipe//mediapipe/calculators/openvino:openvino_inference_calculator",
        #ENABLE_WTIH_OV_CALCULATORS"@mediapipe//mediapipe/calculators/openvino:openvino_tensors_to_classification_calculator",
        #ENABLE_WTIH_OV_CALCULATORS"@mediapipe//mediapipe/calculators/openvino:openvino_tensors_to_detections_calculator",
        "@mediapipe//mediapipe/calculators/core:concatenate_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "@mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:add_header_calculator",
        "@mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:concatenate_vector_calculator_hdr",
        "@mediapipe//mediapipe/calculators/core:concatenate_detection_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:concatenate_proto_list_calculator",
        "@mediapipe//mediapipe/calculators/core:clip_vector_size_calculator",
        #SEGFAULT_ON_GLOG_INIT"@mediapipe//mediapipe/calculators/core:clip_detection_vector_size_calculator",
        "@mediapipe//mediapipe/calculators/core:counting_source_calculator",
        "@mediapipe//mediapipe/calculators/core:make_pair_calculator",
        "@mediapipe//mediapipe/calculators/core:matrix_multiply_calculator",
        "@mediapipe//mediapipe/calculators/core:matrix_subtract_calculator",
        "@mediapipe//mediapipe/calculators/core:mux_calculator",
        "@mediapipe//mediapipe/calculators/core:non_zero_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_cloner_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_inner_join_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",
        "@mediapipe//mediapipe/calculators/core:pass_through_calculator",
        "@mediapipe//mediapipe/calculators/core:round_robin_demux_calculator",
        "@mediapipe//mediapipe/calculators/core:immediate_mux_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_presence_calculator",
        "@mediapipe//mediapipe/calculators/core:string_to_int_calculator",
        "@mediapipe//mediapipe/calculators/core:default_side_packet_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_resampler_calculator",
        "@mediapipe//mediapipe/calculators/core:split_proto_list_calculator",
        "@mediapipe//mediapipe/calculators/core:dequantize_byte_array_calculator",
        "@mediapipe//mediapipe/calculators/core:quantize_float_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:sequence_shift_calculator",
        "@mediapipe//mediapipe/calculators/core:gate_calculator",
        "@mediapipe//mediapipe/calculators/core:matrix_to_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:merge_calculator",
        "@mediapipe//mediapipe/calculators/core:stream_to_side_packet_calculator",
        "@mediapipe//mediapipe/calculators/core:constant_side_packet_calculator",
        "@mediapipe//mediapipe/calculators/core:graph_profile_calculator",
        "@mediapipe//mediapipe/calculators/core:get_vector_item_calculator",
        "@mediapipe//mediapipe/calculators/core:vector_size_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_sequencer_calculator",
        "@mediapipe//mediapipe/calculators/core:merge_to_vector_calculator",
        #TENSORFLOW LITE BELOW
        "@mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "@mediapipe//mediapipe/calculators/image:color_convert_calculator",
        "@mediapipe//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
        "@mediapipe//mediapipe/calculators/image:opencv_image_encoder_calculator",
        "@mediapipe//mediapipe/calculators/image:opencv_put_text_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:set_alpha_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:bilateral_filter_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/image:image_cropping_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:luminance_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:sobel_edges_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:recolor_calculator",
        #REQUIRED_libyuv_DEPENDENCY"@mediapipe//mediapipe/calculators/image:scale_image_calculator",
        "@mediapipe//mediapipe/calculators/image:image_clone_calculator",
        "@mediapipe//mediapipe/calculators/image:image_properties_calculator",
        #GPU"@mediapipe//mediapipe/calculators/image:mask_overlay_calculator",
        "@mediapipe//mediapipe/calculators/image:feature_detector_calculator",
        #REQUIRED_easyexif_DEPENDENCY"@mediapipe//mediapipe/calculators/image:image_file_properties_calculator",
        "@mediapipe//mediapipe/calculators/image:segmentation_smoothing_calculator",
        "@mediapipe//mediapipe/calculators/image:warp_affine_calculator",
        #REQUIRED_libyuv_DEPENDENCY"@mediapipe//mediapipe/calculators/image:yuv_to_image_calculator",
        #VISIBILITY_PRIVATE_THUS_SINK_BELOW"@mediapipe//mediapipe/calculators/internal:callback_packet_calculator",
        "@mediapipe//mediapipe/framework/tool:sink",
        "@mediapipe//mediapipe/calculators/video:flow_to_image_calculator",
        "@mediapipe//mediapipe/calculators/video:tvl1_optical_flow_calculator",
        "@mediapipe//mediapipe/calculators/video:motion_analysis_calculator",
        "@mediapipe//mediapipe/calculators/video:flow_packager_calculator",
        "@mediapipe//mediapipe/calculators/video:box_tracker_calculator",
        "@mediapipe//mediapipe/calculators/video:box_detector_calculator",
        "@mediapipe//mediapipe/calculators/video:tracked_detection_manager_calculator",
        "@mediapipe//mediapipe/calculators/video:video_pre_stream_calculator",
    ],
    linkstatic=1,
)