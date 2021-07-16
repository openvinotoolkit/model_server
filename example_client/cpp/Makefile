#
# Copyright (c) 2021 Intel Corporation
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

.PHONY: default build

default: docker_build

docker_build:
	@docker build -t cpp_clients_build_image:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} .
	@rm -Rf bin
	@mkdir -p bin
	@docker cp $$(docker create --rm cpp_clients_build_image:latest):/build/bazel-bin/src/resnet_client ./bin/resnet_client
