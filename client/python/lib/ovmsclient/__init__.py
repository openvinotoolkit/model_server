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


# Exported API functions

from ovmsclient.tfs_compat.grpc.tensors import make_tensor_proto as make_tensor_proto
from ovmsclient.tfs_compat.grpc.tensors import make_ndarray as make_ndarray
from ovmsclient.tfs_compat.grpc.requests import make_predict_request as make_grpc_predict_request
from ovmsclient.tfs_compat.grpc.requests import make_metadata_request as make_grpc_metadata_request
from ovmsclient.tfs_compat.grpc.requests import make_status_request as make_grpc_status_request
from ovmsclient.tfs_compat.grpc.serving_client import make_grpc_client as make_grpc_client


# Namespaces bindings

class grpcclient(SimpleNamespace):

	make_tensor_proto = make_tensor_proto
	make_ndarray = make_ndarray
	make_predict_request = make_grpc_predict_request
	make_metadata_request = make_grpc_metadata_request
	make_status_request = make_grpc_status_request
	make_client = make_grpc_client
