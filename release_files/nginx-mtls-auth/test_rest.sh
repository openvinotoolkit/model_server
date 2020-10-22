#!/bin/bash -x
python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port 15555 --client_cert client.pem --client_key client.key --ignore_server_verification \
	--model_name face-detection

python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port 15555 --client_cert client.pem --client_key client.key --server_cert ./server.pem \
	--model_name face-detection

