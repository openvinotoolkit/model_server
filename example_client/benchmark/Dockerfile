#
# INTEL CONFIDENTIAL
# Copyright (c) 2021 Intel Corporation
#
# The source code contained or described herein and all documents related to
# the source code ("Material") are owned by Intel Corporation or its suppliers
# or licensors. Title to the Material remains with Intel Corporation or its
# suppliers and licensors. The Material contains trade secrets and proprietary
# and confidential information of Intel or its suppliers and licensors. The
# Material is protected by worldwide copyright and trade secret laws and treaty
# provisions. No part of the Material may be used, copied, reproduced, modified,
# published, uploaded, posted, transmitted, distributed, or disclosed in any way
# without Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellectual
# property right is granted to or conferred upon you by disclosure or delivery
# of the Materials, either expressly, by implication, inducement, estoppel or
# otherwise. Any license under such intellectual property rights must be express
# and approved by Intel in writing.
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
COPY VERSION /ovms_benchmark_client
COPY main.py /ovms_benchmark_client
COPY client.py /ovms_benchmark_client
COPY client_nvtrt.py /ovms_benchmark_client
COPY client_ovms.py /ovms_benchmark_client
COPY metrics.py /ovms_benchmark_client
COPY db_exporter.py /ovms_benchmark_client
COPY db_config.yml /ovms_benchmark_client
COPY __init__.py /ovms_benchmark_client
COPY entrypoint.sh /

RUN mkdir -p /data
COPY road1.jpg /data

RUN mkdir -p /haproxy
COPY haproxy.cfg /haproxy
EXPOSE 11888

RUN chmod +x /ovms_benchmark_client/main.py
ENV PYTHONPATH "${PYTHONPATH}:/ovms_benchmark_client"
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
