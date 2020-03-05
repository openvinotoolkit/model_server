#!/bin/bash
set -ex

STYLE_CHECK_OPTS="${STYLE_CHECK_OPTS:---exclude=ie_serving/tensorflow_serving_api}" 
STYLE_CHECK_DIRS="${STYLE_CHECK_DIRS:-tests ie_serving setup.py}"

python3 -m pip install -U virtualenv
python3 -m virtualenv .venv-style

. .venv-style/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt

flake8 ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}
