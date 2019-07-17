#
# Copyright (c) 2018 Intel Corporation
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
from tensorflow_serving.util import status_pb2


def check_availability_of_requested_model(models, model_name,
                                          requested_version):
    version = 0
    valid_model_spec = False

    try:
        requested_version = int(requested_version)
    except ValueError:
        return valid_model_spec, version

    if model_name in models:
        if requested_version == 0 and models[model_name].default_version != -1:
            version = models[model_name].default_version
            valid_model_spec = True
        elif requested_version in models[model_name].versions:
            version = requested_version
            valid_model_spec = True
    return valid_model_spec, version


def check_availability_of_requested_status(models, model_name,
                                           requested_version):
    try:
        requested_version = int(requested_version)
    except ValueError:
        return False
    if model_name in models:
        if not requested_version:
            return True
        if requested_version in models[model_name].versions_statuses.keys():
            return True
    return False


def add_status_to_response(version_status, response):
    status_proto = status_pb2.StatusProto()
    status_proto.error_code = version_status.status['error_code']
    status_proto.error_message = version_status.status['error_message']
    response.model_version_status.add(version=version_status.version,
                                      state=version_status.state,
                                      status=status_proto)
