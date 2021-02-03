#!/bin/bash
# operator-sdk init --plugins=helm --domain=com --group=intel --version=v1alpha1 --kind=Ovms --helm-chart ../../deploy/ovms/ --helm-chart-version 3.0
export IMG=quay.io/openvino/ovms-operator:v1.0
make docker-build docker-push IMG=$IMG
make install
make deploy IMG=$IMG
# kubectl apply -f config/samples/intel_v1alpha1_ovms.yaml

# create bundle
make bundle


