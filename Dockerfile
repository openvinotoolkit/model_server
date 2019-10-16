FROM ubuntu:18.04
RUN apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            libgomp1 \
            python3-dev \
            python3-pip \
            virtualenv \
            usbutils \
            gnupg2

RUN curl -o GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN echo "deb https://apt.repos.intel.com/openvino/2019/ all main" > /etc/apt/sources.list.d/intel-openvino-2019.list

RUN apt-get update && apt-get install -y intel-openvino-dev-ubuntu18-2019.3.344

ENV PYTHONPATH="/opt/intel/openvino/python/python3.6"
ENV LD_LIBRARY_PATH="/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:/opt/intel/openvino/deployment_tools/inference_engine/external/mkltiny_lnx/lib:/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64"

WORKDIR /ie-serving-py

COPY requirements.txt /ie-serving-py/
RUN virtualenv -p python3 .venv && \
    . .venv/bin/activate && pip3 --no-cache-dir install -r requirements.txt

COPY start_server.sh setup.py version /ie-serving-py/
COPY ie_serving /ie-serving-py/ie_serving

RUN . .venv/bin/activate && pip3 install .
