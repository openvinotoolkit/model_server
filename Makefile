

PY_VERSION := 3
VIRTUALENV_EXE=$(if $(subst 2,,$(PY_VERSION)),python3 -m venv,virtualenv)
VIRTUALENV_DIR=$(if $(subst 2,,$(PY_VERSION)),.venv3,.venv)
ACTIVATE="$(VIRTUALENV_DIR)/bin/activate"
STYLEVIRTUALENV_DIR=".styleenv$(PY_VERSION)"
STYLE_CHECK_OPTS := --exclude=ie_serving/tensorflow_serving_api
STYLE_CHECK_DIRS := tests ie_serving setup.py
TEST_OPTS :=
TEST_DIRS ?= tests/
CONFIG := "$(CONFIG)"
ML_DIR := "$(MK_DIR)"
HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"

.PHONY: default install uninstall requirements \
	venv test unit_test coverage style dist clean \

default: install

venv: $(ACTIVATE)
	@echo -n "Using "
	@. $(ACTIVATE); python3 --version

$(ACTIVATE): requirements.txt requirements-dev.txt
	@echo "Updating virtualenv dependencies in: $(VIRTUALENV_DIR)..."
	@test -d $(VIRTUALENV_DIR) || $(VIRTUALENV_EXE) $(VIRTUALENV_DIR)
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r requirements.txt
	@. $(ACTIVATE); pip$(PY_VERSION) install -qq -r requirements-dev.txt
	@touch $(ACTIVATE)

install:
	@pip install .

run: venv install
	@. $(ACTIVATE); python ie_serving/main.py --config "$CONFIG"

unit_test: venv
	@echo "Running unit tests..."
	@. $(ACTIVATE); py.test $(TEST_OPTS) $(TEST_DIRS)

coverage: venv
	@echo "Computing unit test coverage..."
	@. $(ACTIVATE); py.test --cov-report term-missing --cov=ncloud \
		$(TEST_OPTS) $(TEST_DIRS)

test: venv
	@echo "Executing functional tests..."
	@. $(ACTIVATE); py.test $(TEST_DIRS)/functional/

style: venv
	@echo "Style-checking codebase..."
	@. $(ACTIVATE); flake8 $(STYLE_CHECK_OPTS) $(STYLE_CHECK_DIRS)

clean_pyc:
	@echo "Removing .pyc files..."
	@find . -name '*.pyc' -exec rm -f {} \;

clean: clean_pyc
	@echo "Removing virtual env files..."
	@rm -rf $(VIRTUALENV_DIR)

docker_build:
	@echo "Building docker image"
	@echo build -f Dockerfile --build-arg HTTP_PROXY=$(HTTP_PROXY) --build-arg HTTPS_PROXY="$(HTTPS_PROXY)" -t ie-serving-py:latest .
	@docker build -f Dockerfile --build-arg HTTP_PROXY=$(HTTP_PROXY) --build-arg HTTPS_PROXY="$(HTTPS_PROXY)" -t ie-serving-py:latest .

docker_run:
	@echo "Starting the docker container with serving model"
	@docker run --rm -d --name ie-serving-py-test-multi -v /tmp/test_models/saved_models/:/opt/ml:ro -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
