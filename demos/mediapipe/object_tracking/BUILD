load("@mediapipe//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_simple_subgraph")

mediapipe_simple_subgraph(
    name = "object_detection_ovms_subgraph",
    graph = "subgraphs/object_detection_ovms.pbtxt",
    register_as = "ObjectDetectionSubgraphOVMS",
    deps = [
        "@mediapipe//mediapipe/calculators/ovms:ovms_calculator", # this includes both calculators
        "@mediapipe//mediapipe/calculators/core:concatenate_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "@mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//src:ovmscalculatoroptions_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

mediapipe_simple_subgraph(
    name = "object_tracking_cpu_subgraph",
    graph = "subgraphs/object_tracking_cpu.pbtxt",
    register_as = "ObjectTrackingSubgraphCpu",
    deps = [
        ":box_tracking_cpu_subgraph",
        "@mediapipe//mediapipe/calculators/util:detection_unique_id_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_timed_box_list_calculator",
        "@mediapipe//mediapipe/calculators/video:tracked_detection_manager_calculator",
        "@mediapipe//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
    ],
    visibility = ["//visibility:public"],
)

mediapipe_simple_subgraph(
    name = "box_tracking_cpu_subgraph",
    graph = "subgraphs/box_tracking_cpu.pbtxt",
    register_as = "BoxTrackingSubgraphCpu",
    deps = [
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/video:box_tracker_calculator",
        "@mediapipe//mediapipe/calculators/video:flow_packager_calculator",
        "@mediapipe//mediapipe/calculators/video:motion_analysis_calculator",
        "@mediapipe//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "@mediapipe//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "@mediapipe//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
    ],
    visibility = ["//visibility:public"],
)

mediapipe_simple_subgraph(
    name = "renderer_cpu_subgraph",
    graph = "subgraphs/renderer_cpu.pbtxt",
    register_as = "RendererSubgraphCpu",
    deps = [
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
    visibility = ["//visibility:public"],
)