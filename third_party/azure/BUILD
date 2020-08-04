package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "storage",
    srcs = [
	"Microsoft.WindowsAzure.Storage/build.release/Binaries/libazurestorage.a",
    ],
    hdrs = glob([
	"Microsoft.WindowsAzure.Storage/includes/**/*.h",
	"Microsoft.WindowsAzure.Storage/includes/**/*.dat",
		 ]),
    strip_include_prefix = "Microsoft.WindowsAzure.Storage/includes/",
)
