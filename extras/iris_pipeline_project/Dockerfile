FROM openvino/model_server:latest

USER root

ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python

RUN apt-get update && apt-get install -y python3-pip
RUN pip install --no-cache-dir --break-system-packages \
    pandas numpy scikit-learn joblib skl2onnx onnx onnxruntime \
    scikit-learn-intelex==2025.7.0 \
    tritonclient[all]
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir --break-system-packages \
    intel-extension-for-pytorch oneccl_bind_pt \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

RUN rm /ovms/lib/libtbb.so* && cp /usr/local/lib/libtbb.so* /ovms/lib/

ENTRYPOINT ["/ovms/bin/ovms"]


