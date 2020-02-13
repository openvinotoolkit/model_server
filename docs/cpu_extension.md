# Setting custom Extension Library

## Overview 

Inference engine can be extended by creating custom kernel for network layers.
It might be useful when the graph include layers and operations not supported by default 
by the device plugin.

Implementation of such custom layers can be included in OpenVINO&trade; Model Server for handling the inference
requests.

The process of creating the extension is documented on 
[docs.openvinotoolkit.org](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Integrate_your_kernels_into_IE.html)

The extension should be compiled as a separate library and copied to the OpenVINO Model Server.

OVMS will look for the extension library in the path defined by environment variable `CPU_EXTENSION`.
Without this variable, a [standard set of layers](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html)
 will be supported.


**Note:** The Docker image with OpenVINO Model Server _does not_ include all the tools and sub-components needed to compile the extension library, 
so you might need to execute this process on a separate host.

## Using custom CPU Extension in the Model Server

While the CPU extension is compiled, you can attach it to the docker container with OpenVINO Model Server and reference it be setting its path like in the example:

```bash
docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 --env CPU_EXTENSION=/opt/ml/libcpu_extension.so  ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
```  
