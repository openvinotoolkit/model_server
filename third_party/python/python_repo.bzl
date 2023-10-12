#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

def _python_repository_impl(repository_ctx):
    result = repository_ctx.execute(["find","/", "-iname", "libpython3.*.so"],quiet=False)
    split = result.stdout.partition("\n")
    lib_path = split[0].removeprefix("/usr/")

    result = repository_ctx.execute(["python3","--version"],quiet=False)
    full_version = result.stdout
    split = full_version.removeprefix("Python ").rpartition(".")
    version = split[0]

    build_file_content = """
cc_library(
    name = "python3-lib",
    srcs = ["{lib_path}"],
    hdrs = glob(["include/python{version}/*.h"]),
    includes = ["include/python{version}"],
    visibility = ["//visibility:public"]
)"""
    repository_ctx.file("BUILD", build_file_content.format(version=version, lib_path=lib_path))

python_repository = repository_rule(
    implementation = _python_repository_impl,
    environ = ["BASE_IMAGE"],
    local=False,
)
