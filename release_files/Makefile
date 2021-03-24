#
# Copyright (c) 2020-2021 Intel Corporation
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

.PHONY: centos redhat
all: centos redhat

centos:
	docker build -f Dockerfile.centos -t ovms:centos --build-arg http_proxy=$(http_proxy) --build-arg https_proxy="$(https_proxy)" .

redhat:
	docker build -f Dockerfile.redhat -t ovms:redhat --build-arg http_proxy=$(http_proxy) --build-arg https_proxy="$(https_proxy)" .

