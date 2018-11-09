# Use local version of image built from Dockerfile.cpu in /docker/base
FROM ubuntu:16.04
LABEL maintainer="Intel"

WORKDIR /root

RUN apt-get -y update && \
    apt-get -y install curl && \
    apt-get -y install wget && \
    apt-get -y install vim && \
    apt-get -y install iputils-ping && \
    apt-get -y install nginx && \
    apt-get -y install python-pip && \
    apt-get -y install git

RUN pip --no-cache-dir install tensorflow==1.6.0 tensorflow-serving-api==1.6.1 numpy boto3 six awscli flask==0.11 Jinja2==2.9 gevent gunicorn

RUN cd /tmp && \
    curl -O https://dl.influxdata.com/telegraf/releases/telegraf_1.4.2-1_amd64.deb && \
    dpkg -i telegraf_1.4.2-1_amd64.deb && \
    rm telegraf_1.4.2-1_amd64.deb

# OpenVino installation
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer

ARG DL_INSTALL_DIR=/opt/intel/computer_vision_sdk/deployment_tools
ARG DL_DIR=/tmp

ENV TEMP_DIR TEMP_DIR

RUN apt-get update && apt-get install -y --no-install-recommends cpio cmake \
    python3-pip python3-venv python3-dev python3-setuptools virtualenv

COPY l_openvino_toolkit*.tgz $TEMP_DIR/
RUN cd $TEMP_DIR && pwd && ls && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    pwd | grep -q openvino_toolkit_p ; \
    if [ $? = 0 ];then sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=;intel-ism__noarch;intel-cv-sdk-base-shared__noarch;intel-cv-sdk-base-l-setupvars__noarch;intel-cv-sdk-base-l-inference-engine__noarch;intel-cv-sdk-base-gfx-install__noarch;intel-cv-sdk-base-shared-pset/g' silent.cfg; fi && \
    pwd | grep -q openvino_toolkit_fpga ; \
    if [ $? = 0 ];then sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=;intel-ism__noarch;intel-cv-sdk-full-shared__noarch;intel-cv-sdk-full-l-setupvars__noarch;intel-cv-sdk-full-l-inference-engine__noarch;intel-cv-sdk-full-gfx-install__noarch;intel-cv-sdk-full-shared-pset/g' silent.cfg; fi && \
    ./install.sh -s silent.cfg && \
    rm -Rf $TEMP_DIR $INSTALL_DIR/install_dependencies $INSTALL_DIR/uninstall* /tmp/* $DL_INSTALL_DIR/documentation $DL_INSTALL_DIR/inference_engine/samples

ENV PYTHONPATH="$INSTALL_DIR/python/python3.5/ubuntu16:$INSTALL_DIR/python/python3.5"
ENV LD_LIBRARY_PATH="$DL_INSTALL_DIR/inference_engine/external/cldnn/lib:$DL_INSTALL_DIR/inference_engine/external/gna/lib:$DL_INSTALL_DIR/inference_engine/external/mkltiny_lnx/lib:$DL_INSTALL_DIR/inference_engine/lib/ubuntu_16.04/intel64"

RUN git clone https://github.com/IntelAI/OpenVINO-model-server.git /ie-serving-py

WORKDIR /ie-serving-py

RUN virtualenv -p python3 .venv && \
    . .venv/bin/activate && pip3 --no-cache-dir install -r requirements.txt

RUN . .venv/bin/activate && pip3 install .

# Install sagemaker components

COPY sagemaker_tensorflow_container-1.0.0.tar.gz .

RUN pip install sagemaker_tensorflow_container-1.0.0.tar.gz sagemaker-container-support

RUN rm sagemaker_tensorflow_container-1.0.0.tar.gz

ENTRYPOINT ["entry.py"]

