#!/bin/bash -x
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
	-p 19000:9000 \
	-p 15555:5555 \
	openvino/model_server:rr-ovms-nginx-mtls \
	--model_path /models/face-detection --model_name face-detection --port 9001 --rest_port 5556 --log_level DEBUG --shape auto

