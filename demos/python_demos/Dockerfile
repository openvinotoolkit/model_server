FROM openvino/model_server:latest
USER root
ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python
RUN apt update && apt install -y python3-pip git
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "/ovms/bin/ovms" ]