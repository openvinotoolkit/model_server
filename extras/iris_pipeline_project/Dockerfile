FROM openvino/model_server:latest

USER root

ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --break-system-packages pandas numpy scikit-learn joblib skl2onnx onnx onnxruntime


WORKDIR /workspace
ENTRYPOINT ["/ovms/bin/ovms"]
