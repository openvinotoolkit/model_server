# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.

load("@bazel_skylib//rules:select_file.bzl", "select_file")
load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "sources",
    srcs = glob(["**"]),
)

configure_make(
    name = "vpx_build",
    lib_source = ":sources",
    configure_options = [
        "--target=generic-gnu",
        "--enable-pic",
        "--enable-static",
        "--disable-shared",
        "--disable-vp9",
        "--disable-vp8-encoder",
        "--disable-examples",
        "--disable-tools",
        "--disable-docs",
        "--disable-unit-tests",
    ],
    out_static_libs = ["libvpx.pic.a"],
    postfix_script = 'mv "$$INSTALLDIR$$/lib/libvpx.a" "$$INSTALLDIR$$/lib/libvpx.pic.a"',
)

select_file(
    name = "archive",
    srcs = ":vpx_build",
    subpath = "lib/libvpx.pic.a",
)

cc_library(
    name = "vpx",
    srcs = [":archive"],
    deps = [":vpx_build"],
    linkopts = ["-lm", "-lpthread"],
)
