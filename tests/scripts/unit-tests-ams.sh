#!/bin/bash
set -ex
. .venv-jenkins/bin/activate

pip install -r requirements.txt

pushd .
cd extras/ams_wrapper/tests/unit
./get_test_images.sh
popd
py.test extras/ams_wrapper/tests/unit