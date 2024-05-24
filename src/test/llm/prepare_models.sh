#!/bin/bash

python3 -m venv .venv
. .venv/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
python3 prepare_models.py
