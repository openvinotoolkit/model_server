# Description: Tensorflow Serving examples.

cc_binary(
    name = "resnet_client_cc",
    srcs = [
        "resnet_client.cc",
    ],
    deps = [
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
