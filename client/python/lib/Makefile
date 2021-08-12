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

.PHONY: build-deps build-package build test clean

build-deps:
	@echo "Building dependencies"
	@. $(ACTIVATE); python setup.py build_apis

build-package:
	@echo "Building ovmsclient package"
	@. $(ACTIVATE); python setup.py bdist_wheel

build: build-deps build-package
	@echo "Build completed successfully. The wheel should be present in dist/ directory"

test:
	@echo "Installing requirements"
	@pip install -r tests/requirements.txt
	@echo "Running ovmsclient tests"
	@pytest -v tests/

clean:
	@rm -rf tensorflow/ tensorflow_serving/ build/ dist/
