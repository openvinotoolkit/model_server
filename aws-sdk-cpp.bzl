load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@//third_party/aws:awssdkcpp.bzl", "aws_sdk_cpp_repository")

def aws_workspace():
    # AWS S3 SDK
    native.new_local_repository(
        name = "awssdk",
        build_file = "@//third_party/aws:BUILD",
        path = "/awssdk",
    )

def aws_1_11_111_workspace():
    # TF IO uses 1.7.339
    # we need to match aws-c-common given that aws-sdk-cpp is not built with bazel
    # TODO wrap it into awssdkcpp_deps function
    # check aws-sdk-cpp third-party/CMakeLists.txt" for version
    # for some reason TF IO uses 0.4.29 while the sdk we uses tag
    # / commit ac02e17 -> v0.3.5?
    # TODO DONE check licenses for TF IO apache OK
    # TODO needs aws-c-event-stream
    http_archive(
        name = "aws-c-common",
        build_file = "@//third_party/aws-c-common:BUILD",
        sha256 = "01c2a58553a37b3aa5914d9e0bf7bf14507ff4937bc5872a678892ca20fcae1f",
        strip_prefix = "aws-c-common-0.4.29",
        urls = [
            "https://github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
        ],
    )
    # here we need to be past this commit:
    # 97ab2e5 as this is used in aws-sdk-cpp third-party/CMakeLists.txt 0.1.4 contains this commit
    http_archive(
        name = "aws-c-event-stream",
        build_file = "//third_party/aws-c-event-stream:BUILD",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
    )
    # this is needed by aws-c-event-stream 
    # TODO check what version to use
    http_archive(
        name = "aws-checksums",
        build_file = "//third_party/aws-checksums:BUILD",
        sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
        strip_prefix = "aws-checksums-0.1.5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        ],
    )
    
    http_archive(
        name = "awssdkcpp2",
        build_file = "@//third_party/aws:BUILD",
        strip_prefix = "aws-sdk-cpp-1.11.111",
        urls = [
            #"https://github.com/aws/aws-sdk-cpp/archive/1.7.129.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/refs/tags/1.11.111.tar.gz"
        ],
    )
    new_git_repository(
        name = "aws-sdk-cpp",
        remote = "https://github.com/aws/aws-sdk-cpp.git",
        build_file = "@//third_party/aws:BUILD.bazel",
        commit = "2fcf454a9893fd40cfe3de5aa929521ed7f1b370", # 1.11.111
        init_submodules = True,
    )
    new_git_repository(
        name = "aws-crt-cpp",
        remote = "https://github.com/awslabs/aws-crt-cpp.git",
        build_file = "@//third_party/aws-crt-cpp:BUILD",
        commit = "cb474daeeaf5c025bd3408103adf61b97b74e600", # from aws-sdk-cpp 1.11.111
        init_submodules = True,
    )
def aws_cmake_workspace():
    ###########
    # aws with cmake build
    ###########
    aws_sdk_cpp_repository(name="_aws_sdk_cpp2")
    new_git_repository(
        name = "aws-sdk-cpp",
        remote = "https://github.com/aws/aws-sdk-cpp.git",
        build_file = "@_aws_sdk_cpp2//:BUILD",
        #commit = "2fcf454a9893fd40cfe3de5aa929521ed7f1b370", # 1.11.111
        #tag = "1.7.129", # 1.11.111
        tag = "1.11.268", # issues with ASCI handling of file_test.c *xample file.txt in bazel
        #tag = "1.9.49", # 1.11.111
        init_submodules = True,
        recursive_init_submodules = True,
        patch_cmds = ["find . -name '*xample.txt' -delete"],
    )
    ############
    # end of awsk with cmake
    ###########
def aws_1_7_336_workspace():
    # TF IO uses 1.7.339
    # we need to match aws-c-common given that aws-sdk-cpp is not built with bazel
    # TODO wrap it into awssdkcpp_deps function
    # check aws-sdk-cpp third-party/CMakeLists.txt" for version
    # for some reason TF IO uses 0.4.29 while the sdk we uses tag
    # / commit ac02e17 -> v0.3.5?
    # TODO DONE check licenses for TF IO apache OK
    # TODO needs aws-c-event-stream
    http_archive(
        name = "aws-c-common",
        build_file = "@//third_party/aws-c-common:BUILD",
        sha256 = "01c2a58553a37b3aa5914d9e0bf7bf14507ff4937bc5872a678892ca20fcae1f",
        strip_prefix = "aws-c-common-0.4.29",
        urls = [
            "https://github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
        ],
    )
    # here we need to be past this commit:
    # 97ab2e5 as this is used in aws-sdk-cpp third-party/CMakeLists.txt 0.1.4 contains this commit
    http_archive(
        name = "aws-c-event-stream",
        build_file = "//third_party/aws-c-event-stream:BUILD",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
    )
    # this is needed by aws-c-event-stream 
    # TODO check what version to use
    http_archive(
        name = "aws-checksums",
        build_file = "//third_party/aws-checksums:BUILD",
        sha256 = "6e6bed6f75cf54006b6bafb01b3b96df19605572131a2260fddaf0e87949ced0",
        strip_prefix = "aws-checksums-0.1.5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.5.tar.gz",
        ],
    )
    
    http_archive(
        name = "awssdkcpp2",
        build_file = "@//third_party/aws:BUILD",
        strip_prefix = "aws-sdk-cpp-1.11.111",
        urls = [
            #"https://github.com/aws/aws-sdk-cpp/archive/1.7.129.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/refs/tags/1.11.111.tar.gz"
        ],
    )
    new_git_repository(
        name = "aws-sdk-cpp",
        remote = "https://github.com/aws/aws-sdk-cpp.git",
        build_file = "@//third_party/aws:BUILD.bazel",
        commit = "2fcf454a9893fd40cfe3de5aa929521ed7f1b370", # 1.11.111
        init_submodules = True,
    )
def aws_1_10_57_workspace():
    # TF IO uses 1.7.339
    # we need to match aws-c-common given that aws-sdk-cpp is not built with bazel
    # TODO wrap it into awssdkcpp_deps function
    # check aws-sdk-cpp third-party/CMakeLists.txt" for version
    # for some reason TF IO uses 0.4.29 while the sdk we uses tag
    # / commit ac02e17 -> v0.3.5?
    # TODO DONE check licenses for TF IO apache OK
    # TODO needs aws-c-event-stream
    http_archive(
        name = "aws-c-common",
        build_file = "@//third_party/aws-c-common:BUILD",
        sha256 = "01c2a58553a37b3aa5914d9e0bf7bf14507ff4937bc5872a678892ca20fcae1f",
        strip_prefix = "aws-c-common-0.4.29",
        urls = [
            "https://github.com/awslabs/aws-c-common/archive/v0.4.29.tar.gz",
        ],
    )
    # here we need to be past this commit:
    # 97ab2e5 as this is used in aws-sdk-cpp third-party/CMakeLists.txt 0.1.4 contains this commit
    http_archive(
        name = "aws-c-event-stream",
        build_file = "//third_party/aws-c-event-stream:BUILD",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
    )
    # this is needed by aws-c-event-stream 
    # TODO check what version to use
    http_archive(
        name = "aws-checksums",
        build_file = "//third_party/aws-checksums:BUILD",
        sha256 = "84f226f28f9f97077c924fb9f3f59e446791e8826813155cdf9b3702ba2ec0c5",
        strip_prefix = "aws-checksums-0.1.14",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.14.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.14.tar.gz",
        ],
    )
    
    http_archive(
        name = "aws-sdk-cpp",
        build_file = "//third_party/aws-sdk-cpp_1_10_57:BUILD.bazel",
        strip_prefix = "aws-sdk-cpp-1.10.57",
        urls = [
            #"https://github.com/aws/aws-sdk-cpp/archive/1.7.129.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/refs/tags/1.10.57.tar.gz"
        ],
    )
def aws_1_9_379_workspace():
    # TF IO uses 1.7.339
    # we need to match aws-c-common given that aws-sdk-cpp is not built with bazel
    # TODO wrap it into awssdkcpp_deps function
    # check aws-sdk-cpp third-party/CMakeLists.txt" for version
    # for some reason TF IO uses 0.4.29 while the sdk we uses tag
    # / commit ac02e17 -> v0.3.5?
    # TODO DONE check licenses for TF IO apache OK
    # TODO needs aws-c-event-stream
    http_archive(
        name = "aws-c-common",
        build_file = "@//third_party/aws-c-common:BUILD",
        sha256 = "9d2ea0e6ff0e6e3e93b77a108cfaf514bdfcaa96fb99ffd6368f964b47cf39de",
        strip_prefix = "aws-c-common-0.8.4",
        urls = [
            "https://github.com/awslabs/aws-c-common/archive/v0.8.4.tar.gz",
        ],
    )
    # here we need to be past this commit:
    # 97ab2e5 as this is used in aws-sdk-cpp third-party/CMakeLists.txt 0.1.4 contains this commit
    http_archive(
        name = "aws-c-event-stream",
        build_file = "//third_party/aws-c-event-stream:BUILD",
        sha256 = "31d880d1c868d3f3df1e1f4b45e56ac73724a4dc3449d04d47fc0746f6f077b6",
        strip_prefix = "aws-c-event-stream-0.1.4",
        urls = [
            "https://github.com/awslabs/aws-c-event-stream/archive/v0.1.4.tar.gz",
        ],
    )
    # this is needed by aws-c-event-stream 
    # TODO check what version to use
    http_archive(
        name = "aws-checksums",
        build_file = "//third_party/aws-checksums:BUILD",
        sha256 = "84f226f28f9f97077c924fb9f3f59e446791e8826813155cdf9b3702ba2ec0c5",
        strip_prefix = "aws-checksums-0.1.14",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/v0.1.14.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/v0.1.14.tar.gz",
        ],
    )
    
    new_git_repository(
        name = "aws-sdk-cpp",
        build_file = "//third_party/aws-sdk-cpp_1_9_379:BUILD.bazel",
        remote = "https://github.com/awslabs/aws-sdk-cpp.git",
        #strip_prefix = "aws-sdk-cpp-1.9.379",
        #urls = [
            #"https://github.com/aws/aws-sdk-cpp/archive/1.7.129.tar.gz",
            #    "https://github.com/aws/aws-sdk-cpp/archive/refs/tags/1.9.379.tar.gz"
            #],
        tag = "1.9.379",
        init_submodules = True,
        recursive_init_submodules = True,

    )
    new_git_repository(
        name = "aws-crt-cpp",
        remote = "https://github.com/awslabs/aws-crt-cpp.git",
        build_file = "@//third_party/aws-crt-cpp:BUILD",
        commit = "cb474daeeaf5c025bd3408103adf61b97b74e600", # from aws-sdk-cpp 1.11.111
        init_submodules = True,
    )
