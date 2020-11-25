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

MTLS_IMAGE=${1:-"openvino/model_server:nginx-mtls"}

echo "Loading configuration file from test_config.sh..."
source test_config.sh

if [ -d './images/' ] ; then
        echo "Models are ready - no need to download."
else
        echo "Downloading models..."
        set -e
        ./get_model.sh
        set +e
	echo "Models downloaded."
fi

if [ -f './client.pem' ] ; then
        echo "certificates are ready - no need to generate."
else
        echo "generating certificates..."
        set -e
        ./generate_certs.sh
        set +e
	echo "Certificates are up to date."
fi

echo "Starting container. Hit CTRL+C to stop it. Use another terminal to send some requests, e.g. via using test_rest.sh or test_grpc.sh scripts."
docker run --rm -ti \
        -v $(pwd)/model/:/models/face-detection/1/ \
        -v $(pwd)/server.pem:/certs/server.pem:ro \
        -v $(pwd)/server.key:/certs/server.key:ro \
        -v $(pwd)/client_cert_ca.pem:/certs/client_cert_ca.pem:ro \
        -v $(pwd)/dhparam.pem:/certs/dhparam.pem:ro \
        -v $(pwd)/client_cert_ca.crl:/certs/client_cert_ca.crl:ro \
        -p $REST_PORT:$REST_PORT \
        -p $GRPC_PORT:$GRPC_PORT \
        $MTLS_IMAGE \
        --model_path /models/face-detection --model_name face-detection --grpc_bind_address 8.8.8.8 --port $GRPC_PORT --rest_bind_address 1.1.1.1 --rest_port $REST_PORT --log_level DEBUG --shape auto
#                                                                                           ^^^^^^^                                       ^^^^^^^
#                                                                                           [___________________________________________________}
#                                                                                             those will be replaced by 127.0.0.1 by supervisor.
#

