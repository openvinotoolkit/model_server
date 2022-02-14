
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


#
# THIS FILE HAS BEEN AUTO GENERATED.
#


# External imports

from types import SimpleNamespace


# Exported errors

from ovmsclient.tfs_compat.base.errors import ModelServerError # noqa
from ovmsclient.tfs_compat.base.errors import ModelNotFoundError # noqa
from ovmsclient.tfs_compat.base.errors import InvalidInputError # noqa
from ovmsclient.tfs_compat.base.errors import BadResponseError # noqa


# Exported API functions

from ovmsclient.tfs_compat.grpc.tensors import make_tensor_proto as make_tensor_proto
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client as make_grpc_client
from ovmsclient.tfs_compat.http.serving_client import make_http_client as make_http_client


# Namespaces bindings

class grpcclient(SimpleNamespace):

    make_tensor_proto = make_tensor_proto
    make_client = make_grpc_client


class httpclient(SimpleNamespace):

    make_client = make_http_client
