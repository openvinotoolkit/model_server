#!/bin/bash -x
source test_config.sh

if [ -d './images/' ] ; then
        echo "models are ready"
else
        echo "Downloading models..."
        set -e
        ./get_model.sh
        set +e
fi

python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port $REST_PORT --client_cert client.pem --client_key client.key --ignore_server_verification \
	--model_name face-detection

python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port $REST_PORT --client_cert client.pem --client_key client.key --server_cert ./server.pem \
	--model_name face-detection

