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

FROM haproxy:2.8.2

ARG DEB_BUILD_DEPENDANTS="python3-pip curl"
USER root
### Obtain packages required for .deb creation ###
RUN apt-get update && apt-get install -y ${DEB_BUILD_DEPENDANTS} && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --no-cache -r requirements.txt

ENV TF_CPP_MIN_LOG_LEVEL=2

RUN mkdir /ovms_benchmark_client
COPY main.py /ovms_benchmark_client
COPY client.py /ovms_benchmark_client
COPY client_tfs.py /ovms_benchmark_client
COPY client_kfs.py /ovms_benchmark_client
COPY metrics.py /ovms_benchmark_client
COPY db_exporter.py /ovms_benchmark_client
COPY __init__.py /ovms_benchmark_client
COPY entrypoint.sh /

RUN mkdir /haproxy
COPY haproxy.cfg /haproxy
EXPOSE 11888

RUN chmod +x /ovms_benchmark_client/main.py
ENV PYTHONPATH "${PYTHONPATH}:/ovms_benchmark_client"
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
