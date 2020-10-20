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

ARG BASE_IMAGE=openvino/model_server:latest
FROM $BASE_IMAGE
USER root
RUN	set -e ; \
	set -x ; \
	yum install -y epel-release ;\
        yum install -y nginx wget ; \
        yum clean all ; \
	wget -O /usr/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.2/dumb-init_1.2.2_amd64 ; \
	chmod +x /usr/bin/dumb-init ;

COPY ovms_wrapper /
ENTRYPOINT ["/usr/bin/dumb-init", "--", "bash", "-c", "/ovms_wrapper"]

