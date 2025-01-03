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
export PYTHONPATH=$PYTHONPATH:./../../demos/common/python/
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../../demos/common/python/requirements.txt

python3 ../../demos/face_detection/python/face_detection.py --grpc_port $GRPC_PORT --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results --tls \
	--server_cert server.pem --client_cert client.pem --client_key client.key

