alias(
    name = "opencv",
    actual = select({
        "//src:windows": "@windows_opencv//:opencv",
        "//conditions:default": "@linux_opencv//:opencv",
    }),
)