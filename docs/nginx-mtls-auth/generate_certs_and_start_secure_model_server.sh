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
echo "Building mTLS docker layer..."
./build.sh

if [ -d './images/' ] ; then
        echo "Models are ready - no need to download."
else
        echo "Downloading models..."
        set -e
        ./get_model.sh
        set +e
	echo "Models downloaded."
fi

echo "Generating certificates..."
echo "===================================================================================================================================================="
echo "WARNING: For development and testing only. Please follow your organization security practices on handling and distribution of cryptography material."
echo "===================================================================================================================================================="
openssl req -x509 -nodes -days 1 -newkey rsa:2048 -keyout server.key -out server.pem -subj "/C=US/CN=localhost"
openssl genrsa -out client_cert_ca.key 2048
openssl req -x509 -new -nodes -key client_cert_ca.key -sha256 -days 1024 -out client_cert_ca.pem -subj "/C=US/CN=localhost"
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr -subj "/C=US/CN=client"
openssl x509 -req -in client.csr -CA client_cert_ca.pem -CAkey client_cert_ca.key -CAcreateserial -out client.pem -days 1 -sha256
echo "Key material is ready."
echo "Starting container. Hit CTRL+C to stop it. Use another terminal to send some requests, e.g. via using test_rest.sh or test_grpc.sh scripts."
docker run --rm -ti \
        -v $(pwd)/model/:/models/face-detection/1/ \
        -v $(pwd)/server.pem:/certs/server.pem:ro \
        -v $(pwd)/server.key:/certs/server.key:ro \
        -v $(pwd)/client_cert_ca.pem:/certs/client_cert_ca.pem:ro \
        -p $REST_PORT:$REST_PORT \
        -p $GRPC_PORT:$GRPC_PORT \
        $MTLS_IMAGE \
        --model_path /models/face-detection --model_name face-detection --grpc_bind_address 8.8.8.8 --port $GRPC_PORT --rest_bind_address 1.1.1.1 --rest_port $REST_PORT --log_level DEBUG --shape auto
#                                                                                           ^^^^^^^                                       ^^^^^^^
#                                                                                           [___________________________________________________}
#                                                                                             those will be replaced by 127.0.0.1 by supervisor.
#

