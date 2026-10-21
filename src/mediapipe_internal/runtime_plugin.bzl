# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Link upstream registration objects without copying the host's registries/protos."""

def _runtime_plugin_deps_impl(ctx):
    host_owners = {}
    for host in ctx.attr.provided_by:
        for linker_input in host[CcInfo].linking_context.linker_inputs.to_list():
            host_owners[linker_input.owner] = True

    plugin_inputs = []
    cc_infos = []
    for dep in ctx.attr.deps:
        cc_infos.append(dep[CcInfo])
        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            if linker_input.owner not in host_owners:
                plugin_inputs.append(linker_input)
    return [CcInfo(
        compilation_context = cc_common.merge_cc_infos(cc_infos = cc_infos).compilation_context,
        linking_context = cc_common.create_linking_context(
            # C++ linker contexts use topological order. The default depset
            # order reverses archive traversal, placing proto/util archives
            # before calculators and leaving their symbols unresolved.
            linker_inputs = depset(plugin_inputs, order = "topological"),
        ),
    )]

runtime_plugin_deps = rule(
    implementation = _runtime_plugin_deps_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "provided_by": attr.label_list(providers = [CcInfo]),
    },
)
