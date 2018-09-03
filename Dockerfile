FROM ubuntu:16.04

ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer

ARG DL_INSTALL_DIR=/opt/intel/computer_vision_sdk/deployment_tools
ARG DL_DIR=/tmp

ENV TEMP_DIR $TEMP_DIR

RUN apt-get update && apt-get install -y --no-install-recommends cpio \
    python3-pip python3-venv python3-setuptools virtualenv

COPY l_openvino_toolkit*.tgz $TEMP_DIR/
RUN cd $TEMP_DIR && pwd && ls && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=;intel-ism__noarch;intel-cv-sdk-full-shared__noarch;intel-cv-sdk-full-l-setupvars__noarch;intel-cv-sdk-full-l-inference-engine__noarch;intel-cv-sdk-full-gfx-install__noarch;intel-cv-sdk-full-shared-pset/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -Rf $TEMP_DIR $INSTALL_DIR/install_dependencies $INSTALL_DIR/uninstall* /tmp/* $DL_INSTALL_DIR/documentation $DL_INSTALL_DIR/inference_engine/samples

ENV PYTHONPATH="$INSTALL_DIR/python/python3.5"
ENV LD_LIBRARY_PATH="$DL_INSTALL_DIR/inference_engine/external/cldnn/lib:$DL_INSTALL_DIR/inference_engine/external/gna/lib:$DL_INSTALL_DIR/inference_engine/external/mkltiny_lnx/lib:$DL_INSTALL_DIR/inference_engine/lib/ubuntu_16.04/intel64"

COPY start_server.sh setup.py requirements.txt /ie-serving-py/
COPY ie_serving /ie-serving-py/ie_serving

WORKDIR /ie-serving-py

RUN virtualenv -p python3 .venv && \
    . .venv/bin/activate && pip3 --no-cache-dir install -r requirements.txt

RUN . .venv/bin/activate && pip3 install .
