
package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_binary(
    name = "server_cc",
    srcs = [
        "server.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        #"@org_tensorflow//third_party/eigen3",
        "@com_github_grpc_grpc//:grpc++",
        #"//tensorflow_serving/apis:predict_cc_proto",
        #"@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        #"@org_tensorflow//tensorflow/core:lib",
    ],
)
