#!/bin/bash
set -ex

STYLE_CHECK_OPTS="${STYLE_CHECK_OPTS:---exclude=ie_serving/tensorflow_serving_api --max-line-length 120}" 
STYLE_CHECK_DIRS="${STYLE_CHECK_DIRS:-tests ie_serving setup.py extras}"

. .venv-jenkins/bin/activate

flake8 ${STYLE_CHECK_OPTS} ${STYLE_CHECK_DIRS}
