#!/bin/bash -x

source test_config.sh

./BUILD.sh

if [ -d './images/' ] ; then
        echo "models are ready"
else
        echo "Downloading models..."
        set -e
        ./get_model.sh
        set +e
fi

openssl req -x509 -nodes -days 7 -newkey rsa:2048 -keyout server.key -out server.pem -subj "/C=US/CN=localhost"

openssl genrsa -out client_cert_ca.key 2048
openssl req -x509 -new -nodes -key client_cert_ca.key -sha256 -days 1024 -out client_cert_ca.pem -subj "/C=US/CN=localhost"
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr -subj "/C=US/CN=client"
openssl x509 -req -in client.csr -CA client_cert_ca.pem -CAkey client_cert_ca.key -CAcreateserial -out client.pem -days 7 -sha256


docker run --rm -ti \
	-v $(pwd)/model/:/models/face-detection/1/ \
	-v $(pwd)/server.pem:/certs/server.pem:ro \
	-v $(pwd)/server.key:/certs/server.key:ro \
	-v $(pwd)/client_cert_ca.pem:/certs/client_cert_ca.pem:ro \
	-p $REST_PORT:$REST_PORT \
	-p $GRPC_PORT:$GRPC_PORT \
	openvino/model_server:rr-ovms-nginx-mtls \
	--model_path /models/face-detection --model_name face-detection --grpc_bind_address 8.8.8.8 --port $GRPC_PORT --rest_bind_address 1.1.1.1 --rest_port $REST_PORT --log_level DEBUG --shape auto
#                                                                                           ^^^^^^^                                       ^^^^^^^
#                                                                                           [___________________________________________________}
#                                                                                             those will be replaced by 127.0.0.1 by supervisor.
#

