## Custom NEO Runtime packages

This directory is supposed to contain NEO Runtime packages placed in a separate directory like:

```
drivers
└── <neo_version>
     ├── intel-igc-core_<version>_amd64.deb
     ├── intel-igc-opencl_<version>_amd64.deb
     ├── intel-level-zero-gpu-dbgsym_<version>_amd64.ddeb
     ├── intel-level-zero-gpu_<version>_amd64.deb
     ├── intel-opencl-icd-dbgsym_<version>_amd64.ddeb
     ├── intel-opencl-icd_<version>_amd64.deb
     ├── libigdgmm12_<version>_amd64.deb
     └── libigdgmm12_<version>_amd64.deb
```

With such structure you can build OpenVINO Model Server GPU image with custom NEO Runtime by specifying `INSTALL_DRIVER_VERSION` parameter in `make docker_build` command. The value of `INSTALL_DRIVER_VERSION` should be the name of the subfolder i.e. the `<neo_version>` in above schema. 

**Example:** For packages placed in `drivers/neo` location, the build command could look like this:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
```

```bash
make docker_build BASE_OS=ubuntu OVMS_CPP_DOCKER_IMAGE=ovms_custom_neo INSTALL_DRIVER_VERSION=neo
```
