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

FROM haproxy:2.3.10

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y curl

RUN curl https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install numpy
RUN python3 -m pip install grpcio
RUN python3 -m pip install requests
RUN python3 -m pip install tensorflow==2.*
RUN python3 -m pip install tensorflow-serving-api
RUN python3 -m pip install pymongo
RUN python3 -m pip install retry
RUN python3 -m pip install PyYAML
RUN python3 -m pip install pypng
RUN python3 -m pip install nvidia-pyindex
RUN python3 -m pip install tritonclient[all]

RUN mkdir -p /ovms_benchmark_client
COPY main.py /ovms_benchmark_client
COPY client.py /ovms_benchmark_client
COPY client_ovms.py /ovms_benchmark_client
COPY metrics.py /ovms_benchmark_client
COPY db_exporter.py /ovms_benchmark_client
COPY __init__.py /ovms_benchmark_client
COPY entrypoint.sh /

# Comment it to remove NvTrt support from the image
COPY client_nvtrt.py /ovms_benchmark_client

RUN mkdir -p /data
COPY road1.jpg /data

RUN mkdir -p /haproxy
COPY haproxy.cfg /haproxy
EXPOSE 11888

RUN chmod +x /ovms_benchmark_client/main.py
ENV PYTHONPATH "${PYTHONPATH}:/ovms_benchmark_client"
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
