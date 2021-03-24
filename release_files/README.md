## Building pre-release OVMS docker image

This directory contains an archive containing OpenVINO Model Server binaries, a set of Dockerfiles targetting different distributions, and a friendly Makefile.

In order to build OVMS image, ensure that you have already installed and configured:

- Docker
- GNU make

During a docker image build process, some packages are downloaded from the network.

NOTE: If necessary, ensure that you have a `http_proxy` and `https_proxy` environment variables set. Those are passed to a docker container during the build.

### Build image

CentOS base image:
```bash
make docker_build

```
RedHat base image:
```bash
make docker_build BASE_OS=redhat
```

Image will be tagged ovms:latest.

NOTE: Dockerfiles for other base distributions may be provided. Those work, but were not fully tested at this point.

### Test if image works:

```
docker run --rm ovms:latest --help
```

Usage information shall be displayed.

For further information on how to use this container, please read [usage documentation](user-guide/README.md).

