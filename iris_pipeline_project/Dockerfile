FROM openvino/model_server:latest

USER root

ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python

RUN apt update && apt install -y python3-pip git \
    build-essential python3-dev libatlas-base-dev

RUN pip3 install --break-system-packages numpy pandas scikit-learn


COPY pipeline/graph.pbtxt /models/iris_pipeline/
COPY pipeline/ovmsmodel.py /models/iris_pipeline/
COPY model/model.pkl /models/iris_pipeline/
COPY model_config.json /model_config.json

ENTRYPOINT [ "/ovms/bin/ovms" ]
