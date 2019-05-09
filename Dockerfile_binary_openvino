FROM ubuntu:16.04

ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

ARG DL_INSTALL_DIR=/opt/intel/openvino/deployment_tools
ARG DL_DIR=/tmp
ARG https_proxy

ENV TEMP_DIR $TEMP_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
    cpio \
    python3-dev \
    python3-pip \
    python3-venv \
    usbutils \
    virtualenv

ADD l_openvino_toolkit*.tgz $TEMP_DIR/
RUN cd $TEMP_DIR/l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    case $(pwd) in \
        *openvino_toolkit_p_2019*) \
            COMPONENTS='intel-openvino-base__noarch;intel-openvino-dldt-base__noarch;intel-openvino-setupvars__x86_64;intel-openvino-ie-sdk-ubuntu-xenial__x86_64;intel-openvino-ie-rt__x86_64;intel-openvino-ie-rt-core-ubuntu-xenial__x86_64;intel-openvino-ie-rt-cpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-vpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-hddl-ubuntu-xenial__x86_64;intel-openvino-gfx-driver__x86_64;intel-openvino-base-pset' \
            ;; \
        *openvino_toolkit_fpga_2019*) \
            COMPONENTS='intel-openvino-full__noarch;intel-openvino-dldt-full__noarch;intel-openvino-setupvars__x86_64;intel-openvino-ie-sdk-ubuntu-xenial__x86_64;intel-openvino-ie-rt__x86_64;intel-openvino-ie-rt-core-ubuntu-xenial__x86_64;intel-openvino-ie-rt-cpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-vpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-hddl-ubuntu-xenial__x86_64;intel-openvino-gfx-driver__x86_64;intel-openvino-full-pset' \
            ;; \
        *) \
            COMPONENTS=DEFAULTS \
            ;; \
    esac ; \
    sed -i "s/COMPONENTS=DEFAULTS/COMPONENTS=$COMPONENTS/g" silent.cfg && \    
    ./install.sh -s silent.cfg --ignore-signature && \
    rm -Rf $TEMP_DIR $INSTALL_DIR/install_dependencies $INSTALL_DIR/uninstall* /tmp/* $DL_INSTALL_DIR/documentation $DL_INSTALL_DIR/inference_engine/samples

ENV PYTHONPATH="$INSTALL_DIR/python/python3.5"
ENV LD_LIBRARY_PATH="$DL_INSTALL_DIR/inference_engine/external/tbb/lib:$DL_INSTALL_DIR/inference_engine/external/mkltiny_lnx/lib:$DL_INSTALL_DIR/inference_engine/lib/intel64"

WORKDIR /ie-serving-py

COPY start_server.sh setup.py version requirements.txt /ie-serving-py/
RUN virtualenv -p python3 .venv && . .venv/bin/activate && pip3 install -r requirements.txt

COPY ie_serving /ie-serving-py/ie_serving

RUN . .venv/bin/activate && pip3 install .
