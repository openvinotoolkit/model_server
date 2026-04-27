#
# Copyright (c) 2026 Intel Corporation
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

import itertools
import pytest

from tests.functional.utils.inference.communication import GRPC, REST
from tests.functional.utils.inference.inference_client_factory import InferenceClientFactory
from tests.functional.utils.inference.serving import KFS, OPENAI, TFS, TRITON, COHERE
from tests.functional.constants.ovms_type import OvmsType

from ovms import config as ovms_config


def api_type_non_fixture(serving, communication, ovms_type=None):
    return InferenceClientFactory.get_client(serving=serving, communication=communication, ovms_type=ovms_type)


_possible_api_types = list(itertools.product([TFS, KFS], [GRPC, REST]))
if OvmsType.CAPI in ovms_config.ovms_types:
    _possible_api_types += [OvmsType.CAPI]


@pytest.fixture(scope="session", params=_possible_api_types, ids=lambda x: f":".join(x).upper() if len(x) == 2 else x)
def api_type(request):
    if request.param == OvmsType.CAPI:
        return api_type_non_fixture(serving=None, communication=None, ovms_type=request.param)
    else:
        return api_type_non_fixture(*request.param, ovms_type=None)


@pytest.fixture(scope="session", params=itertools.product([TFS], [GRPC, REST]), ids=lambda x: f":".join(x).upper())
def tfs_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(TFS, REST)], ids=lambda x: f":".join(x).upper())
def tfs_rest_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(TFS, GRPC)], ids=lambda x: f":".join(x).upper())
def tfs_grpc_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(KFS, GRPC)], ids=lambda x: f":".join(x).upper())
def kfs_grpc_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(KFS, REST)], ids=lambda x: f":".join(x).upper())
def kfs_rest_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=itertools.product([KFS], [GRPC, REST]), ids=lambda x: f":".join(x).upper())
def kfs_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(OPENAI, REST)], ids=lambda x: f":".join(x).upper())
def openai_rest_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(COHERE, REST)], ids=lambda x: f":".join(x).upper())
def cohere_rest_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=itertools.product([TRITON], [GRPC, REST]), ids=lambda x: f":".join(x).upper())
def triton_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(TRITON, GRPC)], ids=lambda x: f":".join(x).upper())
def triton_grpc_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=[(TRITON, REST)], ids=lambda x: f":".join(x).upper())
def triton_rest_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=itertools.product([KFS, TFS], [REST]), ids=lambda x: f":".join(x).upper())
def rest_api_type(request):
    return api_type_non_fixture(*request.param)


@pytest.fixture(scope="session", params=itertools.product([KFS, TFS], [GRPC]), ids=lambda x: f":".join(x).upper())
def grpc_api_type(request):
    return api_type_non_fixture(*request.param)
