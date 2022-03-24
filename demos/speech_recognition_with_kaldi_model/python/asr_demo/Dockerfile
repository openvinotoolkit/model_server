#
# Copyright (c) 2021-2022 Intel Corporation
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

FROM debian:10
LABEL maintainer="rick@scriptix.io"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        python3 \
        zlib1g-dev \
        ca-certificates \
        gfortran \
        patch \
        ffmpeg \
	vim && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi #EOL
RUN    cd /opt/kaldi/tools && \
       ./extras/install_mkl.sh && \
       make -j $(nproc) && \
       cd /opt/kaldi/src && \
       ./configure --shared && \
       make depend -j $(nproc) && \
       make -j $(nproc) && \
       find /opt/kaldi -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
       find /opt/intel -type f -name "*.a" -exec rm {} \; && \
       find /opt/intel -type f -regex '.*\(_mc.?\|_mic\|_thread\|_ilp64\)\.so' -exec rm {} \; && \
       rm -rf /opt/kaldi/.git

RUN cd /opt/kaldi/egs/aspire/s5 && \
    wget https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2 && \
    tar -xvf 0001_aspire_chain_model_with_hclg.tar.bz2 && \
    rm -f 0001_aspire_chain_model_with_hclg.tar.bz2

RUN apt-get install -y virtualenv

RUN git clone -b stateful_client_extension https://github.com/openvinotoolkit/model_server.git /opt/model_server && \
    cd /opt/model_server && \
    virtualenv -p python3 .venv && \
    . .venv/bin/activate && \
    pip install tensorflow-serving-api==2.* kaldi-python-io==1.2.1 && \
    echo "source /opt/model_server/.venv/bin/activate" | tee -a /root/.bashrc && \
    mkdir /opt/workspace

WORKDIR /opt/workspace/

