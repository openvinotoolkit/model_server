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

VIRTUALENV_EXE := python3 -m virtualenv -p python3
VIRTUALENV_DIR := .venv
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"

.PHONY: build-deps install-package test clean install

venv:$(ACTIVATE)
	@echo -n "Using virtual environment: $(VIRTUALENV_DIR)\n"
	@. $(ACTIVATE); python3 --version

$(ACTIVATE):
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip install --upgrade pip
	@. $(ACTIVATE); pip install -r tests/requirements.txt
	@touch $(ACTIVATE)


build-deps: venv
	@echo "Building dependencies"
	@sudo apt install -y protobuf-compiler
	@. $(ACTIVATE); pip install setuptools
	@. $(ACTIVATE); python setup.py build_apis

install-package: venv
	@echo "Building and installing ovmsclient package"
	@. $(ACTIVATE); python setup.py bdist_wheel
	@. $(ACTIVATE); pip install dist/ovmsclient-0.1-py3-none-any.whl

reinstall-package: venv
	@echo "Rebuilding and reinstalling ovmsclient package"
	@. $(ACTIVATE); python setup.py bdist_wheel
	@. $(ACTIVATE); pip install dist/ovmsclient-0.1-py3-none-any.whl --force-reinstall --no-deps

test: venv
	@echo "Running ovmsclient tests"
	@. $(ACTIVATE); pytest -v tests/

clean:
	@rm -rf tensorflow/ tensorflow_serving/ build/ dist/

install: build-deps install-package test clean
	@echo "Installation successful"

