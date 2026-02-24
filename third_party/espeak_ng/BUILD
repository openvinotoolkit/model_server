
# third_party/espeak_ng/BUILD

config_setting(
    name = "is_macos",
    values = {"cpu": "darwin"},
)

cc_library(
    name = "espeak_ng",
    copts = select({
        ":is_macos": [
            # Adjust to where Homebrew (or your installer) puts headers
            "-I" + "$(HOME)/.brew/opt/espeak-ng/include",
            "-I" + "$(HOME)/.brew/opt/espeak-ng/include/espeak-ng",
        ],
        "//conditions:default": [
            # Typical on Debian/Ubuntu when installing libespeak-ng-dev
            "-I/usr/include",
            "-I/usr/include/espeak-ng",
        ],
    }),
    linkopts = select({
        ":is_macos": [
            "-L" + "$(HOME)/.brew/opt/espeak-ng/lib",
            "-lespeak-ng",
        ],
        "//conditions:default": ["-lespeak-ng"],
    }),
    visibility = ["//visibility:public"],
)
