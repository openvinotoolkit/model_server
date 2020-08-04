package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "sdk",
    srcs = [
        "Release/build.release/Binaries/libcpprest.a",
    ],
    hdrs = glob([
	"Release/include/**/*.h",
	"Release/include/**/*.dat",
		 ]),
    strip_include_prefix = "Release/include",
)

