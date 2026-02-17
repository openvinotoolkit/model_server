#!/bin/bash
docker build -f Dockerfile_bfcl -t bfcl-unary:latest .
docker build --build-arg MODE=stream -f Dockerfile_bfcl -t bfcl-stream:latest .
