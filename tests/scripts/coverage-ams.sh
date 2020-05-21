#!/bin/bash
set -ex
. .venv-jenkins/bin/activate

pip install -r requirements.txt

pushd .
cd extras/ams_wrapper/tests/unit
./get_test_images.sh
popd
pytest --cov-config=extras/ams_wrapper/.coveragerc --cov=src extras/ams_wrapper/tests/ --cov-report=html --cov-fail-under=70
