from ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
run apt-get update && apt-get install -y git software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
python3.6 python3.7 python3.8 python3.9 python3.10 python3.11 \
python3.6-distutils python3.7-distutils python3.8-distutils python3.9-distutils python3.10-distutils python3.11-distutils \
python3.6-venv python3.7-venv python3.8-venv python3.9-venv python3.10-venv python3.11-venv \
python3-pip
copy client /model_server/client
copy demos /model_server/demos
run pip install invoke
copy tests/python/tasks.py .
run invoke deps --verbose --fastFail
entrypoint ["invoke", "test"]
