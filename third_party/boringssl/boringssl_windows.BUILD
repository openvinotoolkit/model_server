cc_library(
    name = "ssl",
    hdrs = glob(["include/openssl/*"]),
    srcs = glob(["lib/ssl.lib"]),
    copts = ["-lcrypto", "-lssl"],
    visibility = ["//visibility:public"],
)
