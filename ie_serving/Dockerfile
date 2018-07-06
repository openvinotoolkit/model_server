FROM ubuntu:16.04

ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/13232/l_openvino_toolkit_fpga_p_2018.2.300_online.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer

ARG DL_INSTALL_DIR=/opt/intel/computer_vision_sdk/deployment_tools
ARG DL_DIR=/tmp

ENV TEMP_DIR TEMP_DIR

RUN apt-get update && apt-get install -y --no-install-recommends wget cpio cmake sudo \
    python3-pip python3-venv python3-setuptools virtualenv

RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && wget -c $DOWNLOAD_LINK && \
    pwd && ls && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=;intel-ism__noarch;intel-cv-sdk-full-shared__noarch;intel-cv-sdk-full-l-setupvars__noarch;intel-cv-sdk-full-l-model-optimizer__noarch;intel-cv-sdk-full-l-inference-engine__noarch;intel-cv-sdk-full-gfx-install__noarch;intel-cv-sdk-full-shared-pset/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -Rf /TEMP_DIR $INSTALL_DIR/install_dependencies $INSTALL_DIR/uninstall* /tmp/* $DL_INSTALL_DIR/documentation $DL_INSTALL_DIR/inference_engine/samples

ENV PYTHONPATH="$INSTALL_DIR/python/python3.5:$DL_INSTALL_DIR/model_optimizer"
ENV LD_LIBRARY_PATH="$DL_INSTALL_DIR/inference_engine/external/cldnn/lib:\
    $DL_INSTALL_DIR/inference_engine/external/gna/lib:\
    $DL_INSTALL_DIR/inference_engine/external/mkltiny_lnx/lib:\
    $DL_INSTALL_DIR/inference_engine/lib/ubuntu_16.04/intel64"

COPY . /ie-serving

WORKDIR /ie-serving

RUN virtualenv -p python3 .venv && \
    . .venv/bin/activate && \
    pip3 --no-cache-dir install -r requirements.txt
#RUN make install

# Set path to serving config - it will be used by make run target
ARG CONFIG=/opt/ml/config.json
ARG MODEL=/opt/ml/ir_model

#ENTRYPOINT make run


