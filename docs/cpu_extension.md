# Setting custom CPU extension library

## Overview 

CPU extension is an integral part of Inference Engine. This library contains code of graph layers which are not a part
of CPU plugin. 

By default OpenVINO&trade; Model server is using the library location in
 `/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so`
 
You might need to change this value by setting environment variable `CPU_EXTENSION` to match the correct path in the following cases:
* you installed Intel OpenVINO&trade; in non default path which is `/opt/intel/computer_vision_sdk`.
* you use non ubuntu OS to host `ie-serving-py` service.

There might be also situations when you need to recompile this library:
* your HW does not support AVX2 CPU feature.
* you would like to take advantage of all CPU optimization features like AVX-512.
* you would like to add support for extra layer types not supported out-of-the-box.

## Compiling CPU extension

When you compile the entire list of the OpenVINO&trade; samples via 
`/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/build_samples.sh`,
 the `cpu_extension` library is also compiled. 

For performance, the library's `cmake` script detects your computer configuration and enables platform optimizations. 

Alternatively, you can explicitly use `cmake` flags: `-DENABLE_AVX2=ON, -DENABLE_AVX512F=ON or -DENABLE_SSE42=ON` when cross-compiling this library for another platform.

More information about customizing the CPU extensions is in
 [OpenVINO&trade; documentation](https://software.intel.com/en-us/articles/OpenVINO-InferEngine#Adding%20your%20own%20kernels) 
 
 
**Note:** The docker image with OpenVINO&trade; model server does not include all the tools and sub-components needed to recompile the CPU extension, so you might need to execute this process on a separate host.

## Using custom CPU extension in the model server

While the CPU extension is recompiled, you can attach it to the docker container with OpenVINO&trade; model server and reference it be setting its path like in the example:

```bash
docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 --env CPU_EXTENSION=/opt/ml/libcpu_extension.so  ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
```  
