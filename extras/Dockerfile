FROM openvino/model_server:latest

USER root

ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python

COPY model/iris_logreg /models/iris_logreg
COPY pipeline/graph.pbtxt /models/iris_pipeline/
COPY pipeline/ovmsmodel.py /models/iris_pipeline/
COPY model_config.json /model_config.json

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --break-system-packages pandas numpy scikit-learn joblib skl2onnx onnx

ENTRYPOINT ["/ovms/bin/ovms", "--config_path", "/model_config.json"]