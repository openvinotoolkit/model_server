#
# Copyright (c) 2021 Intel Corporation
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

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

# Import all modules for ovmsclient_export discovery
from ovmsclient.tfs_compat.grpc import requests, responses, serving_client, tensors
from ovmsclient.custom import management_client

from ovmsclient.util.ovmsclient_export import NAME_TO_SYMBOL_MAPPING, MAIN_NAMESPACE_NAME

LICENCE_PREAMBULE = """
#
# Copyright (c) 2021 Intel Corporation
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
"""

AUTOGEN_WARNING = """
# 
# THIS FILE HAS BEEN AUTO GENERATED.
#
"""

EXTERNAL_IMPORTS = """
# External imports

from types import SimpleNamespace

"""

SEPARATOR = "\n"

def main():
    contents = [LICENCE_PREAMBULE, AUTOGEN_WARNING, EXTERNAL_IMPORTS]

    import_template = "from {} import {} as {}"
    imports = ["# Exported API functions\n"]
    bindings_template = "\t{} = {}"
    namespaces = ["# Namespaces bindings\n"]

    for namespace_name, functions in NAME_TO_SYMBOL_MAPPING.items():
        if namespace_name == MAIN_NAMESPACE_NAME:
            for main_function_name, function in functions.items():
                _, function_impl = function
                imports.append(import_template.format(function_impl.__module__, function_impl.__name__, main_function_name))
            imports.append(SEPARATOR)
        else:
            namespace = ["class {}(SimpleNamespace):\n".format(namespace_name)]
            bindings = []
            for namespaced_function_name, function in functions.items():
                main_function_name, function_impl = function
                bindings.append(bindings_template.format(namespaced_function_name, main_function_name))
            namespace.extend([*bindings, SEPARATOR])
            namespaces.extend(namespace)

    contents.extend([*imports, *namespaces])
    contents_str = SEPARATOR.join(contents)

    with open("__init__.py", "w") as init_file:
        init_file.write(contents_str)

if __name__ == "__main__":
    main()
