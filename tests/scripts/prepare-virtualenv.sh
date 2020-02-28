#!/bin/bash
set -ex

# Check if pip is installed
python3 -m pip --version || bash -c "echo pip3 is not installed; exit 1"

python3 -m pip install -U virtualenv
python3 -m virtualenv .venv

source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
