#
# Copyright (c) 2023 Intel Corporation
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

.PHONY: setup

default: setup

setup:
	mkdir dataset
	wget https://huggingface.co/datasets/clip-benchmark/wds_vtab-cifar10/resolve/main/test/0.tar -O dataset/0.tar
	wget https://huggingface.co/datasets/clip-benchmark/wds_vtab-cifar10/resolve/main/test/1.tar -O dataset/1.tar
	wget https://huggingface.co/datasets/clip-benchmark/wds_vtab-cifar10/resolve/main/test/2.tar -O dataset/2.tar
	wget https://huggingface.co/datasets/clip-benchmark/wds_vtab-cifar10/resolve/main/test/3.tar -O dataset/3.tar
	cd dataset && tar -xvf 0.tar && tar -xvf 1.tar && tar -xvf 2.tar && tar -xvf 3.tar && cd ..

clean:
	@rm -rf dataset
