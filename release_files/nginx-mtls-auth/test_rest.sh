#!/bin/bash -x
python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key \
	--model_name face-detection
