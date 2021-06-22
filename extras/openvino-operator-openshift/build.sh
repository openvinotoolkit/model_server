#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Operator content generated via
# operator-sdk init --plugins=helm --domain=com --group=intel --version=v1alpha1 --kind=Ovms --helm-chart ../../deploy/ovms/ --helm-chart-version 4.0
export IMG=quay.io/openvino/ovms-operator:0.2.0
make docker-build docker-push IMG=$IMG




