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

RUN curl -o GPG-PUB-KEY-INTEL-OPENVINO-2020 https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020
RUN apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020
RUN echo "deb https://apt.repos.intel.com/openvino/2020/ all main" > /etc/apt/sources.list.d/intel-openvino-2020.list

RUN apt-get update && apt-get install -y intel-openvino-dev-ubuntu18-2020.2.130

ENV DL_INSTALL_DIR=/opt/intel/openvino/deployment_tools
ENV PYTHONPATH="/opt/intel/openvino/python/python3.6"
ENV LD_LIBRARY_PATH="$DL_INSTALL_DIR/inference_engine/external/tbb/lib:$DL_INSTALL_DIR/inference_engine/external/mkltiny_lnx/lib:$DL_INSTALL_DIR/inference_engine/lib/intel64:$DL_INSTALL_DIR/ngraph/lib"

WORKDIR /ie-serving-py

COPY requirements.txt /ie-serving-py/
RUN virtualenv -p python3 .venv && \
    . .venv/bin/activate && pip3 --no-cache-dir install -r requirements.txt

COPY start_server.sh setup.py version /ie-serving-py/
COPY ie_serving /ie-serving-py/ie_serving

RUN . .venv/bin/activate && pip3 install .
