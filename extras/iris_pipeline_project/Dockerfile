FROM openvino/model_server:latest

USER root

ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python

RUN apt-get update && apt-get install -y python3-pip
RUN rm /ovms/lib/libtbb.so* && cp /usr/local/lib/libtbb.so* /ovms/lib/
RUN pip install --break-system-packages pandas numpy scikit-learn==1.3.2 joblib skl2onnx onnx onnxruntime scikit-learn-intelex==2024.0.0

ENTRYPOINT ["/ovms/bin/ovms"]
