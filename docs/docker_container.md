# Using Docker Containers {#ovms_docs_docker_container}
_
## Introduction

OpenVINO&trade; Model Server is a serving system for machine learning models. OpenVINO&trade; Model Server makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. This guide will help you deploy OpenVINO&trade; Model Server through docker containers.

## System Requirements

### Hardware 
* Required:
    * 6th to 11th generation Intel® Core™ processors and Intel® Xeon® processors.
* Optional:
    * Intel® Neural Compute Stick 2.
    * Intel® Iris® Pro & Intel® HD Graphics
    * Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.

## Overview 

This guide provides step-by-step instructions on how to deploy OpenVINO&trade; Model Server for Linux using Docker Container including a Quick Start guide. Links are provided for different compatible hardwares. Following instructions are covered in this:

- <a href="#quickstart">Quick Start Guide for OpenVINO&trade; Model Server</a>
- <a href="#sourcecode">Building the OpenVINO&trade; Model Server Image </a>
- <a href="#singlemodel">Starting Docker Container with a Single Model
- <a href="#configfile">Starting Docker container with a configuration file for multiple models</a>
- <a href="#params">Configuration Parameters</a>
- <a href="#storage">Cloud Storage Requirements</a>
- <a href="#ai">Running OpenVINO&trade; Model Server with AI Accelerators NCS, HDDL and GPU</a>
- <a href="#sec">Security Considerations</a>


## Quick Start Guide <a name="quickstart"></a>

A quick start guide to download models and run OpenVINO&trade; Model Server is provided below. 
It allows you to setup OpenVINO&trade; Model Server and run a Face Detection Example.

Refer [Quick Start guide](./ovms_quickstart.md) to set up OpenVINO&trade; Model Server.


## Detailed steps to deploy OpenVINO&trade; Model Server using Docker container

### Install Docker

Install Docker using the following link:

- [Install Docker Engine](https://docs.docker.com/engine/install/)

### Pulling OpenVINO&trade; Model Server Image

After Docker installation you can pull the OpenVINO&trade; Model Server image. Open Terminal and run following command:

```bash
docker pull openvino/model_server:latest
```

Alternatively pull the image from [RedHat Ecosystem Catalog](https://catalog.redhat.com/software/containers/intel/openvino-model-server/607833052937385fc98515de)
```bash
docker pull registry.connect.redhat.com/intel/openvino-model-server:latest
```

###  Building the OpenVINO&trade; Model Server Docker Image<a name="sourcecode"></a>

<details><summary>Building a Docker image</summary>


To build your own image, use the following command in the [git repository root folder](https://github.com/openvinotoolkit/model_server), 

```bash
   make docker_build
```

It will generate the images, tagged as:

- openvino/model_server:latest - with CPU, NCS and HDDL support,
- openvino/model_server-gpu:latest - with CPU, NCS, HDDL and iGPU support,
- openvino/model_server:latest-nginx-mtls - with CPU, NCS and HDDL support and a reference nginx setup of mTLS integration,
as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.
</details>

*Note:* Latest images include OpenVINO 2021.4 release.

*Note:* OVMS docker image could be created with ubi8-minimal base image, centos7 or the default ubuntu20.
Use command `make docker_build BASE_OS=redhat` or `make docker_build BASE_OS=centos`. OVMS with ubi base image doesn't support NCS and HDDL accelerators.

Additionally you can set version of GPU driver used by the produced image. Currently following versions are available:
- 19.41.14441
- 20.35.17767

Provide version from the list above as INSTALL_DRIVER_VERSION argument in make command to build image with specific version of the driver like 
`make docker_build INSTALL_DRIVER_VERSION=19.41.14441`. 
If not provided, version 20.35.17767 is used.












