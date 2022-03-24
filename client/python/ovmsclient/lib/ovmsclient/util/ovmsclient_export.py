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

import functools
from collections import defaultdict

MAIN_NAMESPACE_NAME = "main"
NAME_TO_SYMBOL_MAPPING = defaultdict(dict)


class api_export(object):
    """Provides ways to export symbols to the ovmsclient API."""

    def __init__(self, name, **kwargs):
        self.main_name = name
        self.namespaced_names = kwargs

    def __call__(self, func):
        # Create mapping for main namespace
        NAME_TO_SYMBOL_MAPPING[MAIN_NAMESPACE_NAME][self.main_name] = (self.main_name, func)

        # Create mapping for additional namespaces
        for namespace, namespaced_name in self.namespaced_names.items():
            NAME_TO_SYMBOL_MAPPING[namespace][namespaced_name] = (self.main_name, func)

        return func


ovmsclient_export = functools.partial(api_export)
