#!/bin/bash -x
python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key --ignore_server_verification \
	--model_name face-detection

python3 ../../example_client/rest_get_model_status.py \
	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key --ca ./server.pem \
	--model_name face-detection

#python3 ../../example_client/rest_get_model_status.py \
#	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key --ca ./server.pem \#
#	 --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name data --output_name prob --rest_port 8000 --transpose_input False

# --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results
