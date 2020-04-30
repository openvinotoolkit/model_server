#!/bin/bash
set -ex
. .venv-jenkins/bin/activate

pushd .
cd extras/ams_wrapper/tests
./get_test_images.sh
popd
py.test extras/ams_wrapper/tests