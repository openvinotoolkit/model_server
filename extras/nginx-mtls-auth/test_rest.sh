#!/bin/bash -x
#
# Copyright (c) 2020 Intel Corporation
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


source test_config.sh

if [ -d './images/' ] ; then
        echo "models are ready"
else
        echo "Downloading models..."
        set -e
        ./get_model.sh
        set +e
fi

python3 ../../client/python/tensorflow-serving-api/samples/rest_get_model_status.py \
	--rest_url https://localhost --rest_port $REST_PORT --client_cert client.pem --client_key client.key --ignore_server_verification \
	--model_name face-detection

python3 ../../client/python/tensorflow-serving-api/samples/rest_get_model_status.py \
	--rest_url https://localhost --rest_port $REST_PORT --client_cert client.pem --client_key client.key --server_cert ./server.pem \
	--model_name face-detection

