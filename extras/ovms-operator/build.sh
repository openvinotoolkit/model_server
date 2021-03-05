#!/bin/bash
# Operator content generated via
# operator-sdk init --plugins=helm --domain=com --group=intel --version=v1alpha1 --kind=Ovms --helm-chart ../../deploy/ovms/ --helm-chart-version 3.0
export IMG=quay.io/openvino/ovms-operator:0.1.0
make docker-build docker-push IMG=$IMG




