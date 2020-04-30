#!/bin/bash
set -ex
. .venv-jenkins/bin/activate

py.test extras/ams_wrapper/tests