#!/bin/bash -x
#python3 ../../example_client/rest_get_model_status.py \
#	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key --ignore_server_verification \
#	--model_name face-detection

#python3 ../../example_client/rest_get_model_status.py \
#	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key --ca ./server.pem \
#	--model_name face-detection

#python3 ../../example_client/rest_get_model_status.py \
#	--rest_url https://localhost --rest_port 15555 --client_certificate_file client.pem --client_key_file client.key --ca ./server.pem \
#	 --images_numpy_path imgs.npy --labels_numpy_path lbs.npy --input_name data --output_name prob --rest_port 8000 --transpose_input False


#parser.add_argument('--tls', default=False, action='store_true', help='use TLS communication with gRPC endpoint')
#parser.add_argument('--server_cert', required=False, help='Path to server certificate')
#parser.add_argument('--client_cert', required=False, help='Path to client certificate')
#parser.add_argument('--client_key', required=False, help='Path to client key')

python3 ../../example_client/face_detection.py --grpc_port 19000 --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results --tls \
	--server_cert /home/rr/GH_OVMS/inference-model-manager/scripts/ojej/ca-cert-tf.crt \
	--client_cert /home/rr/GH_OVMS/inference-model-manager/scripts/ojej/client-tf.crt \
	--client_key /home/rr/GH_OVMS/inference-model-manager/scripts/ojej/client-tf.key

# --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results
