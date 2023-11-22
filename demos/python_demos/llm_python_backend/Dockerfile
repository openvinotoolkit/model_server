FROM ovms-21-11-2023-dk:milosz  # TODO: Use Milosz image ?
USER root
ENV LD_LIBRARY_PATH=/ovms/lib
ENV PYTHONPATH=/ovms/lib/python
RUN apt update && apt install -y python3-pip git
RUN pip3 install git+https://github.com/huggingface/optimum-intel.git onnx
