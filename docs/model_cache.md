# Model Cache {#ovms_docs_model_cache}

## Overview
The Model Server can leverage a [OpenVINO&trade; model cache functionality](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Model_caching_overview.html), to speed up subsequent model loading on a target device. 
The cached files make the Model Server initialization usually faster. 
The boost depends on a model and a target device. The most noticeable improvement will be observed with GPU devices. On other devices, like CPU, it is possible to observe no speed up effect or even slower loading process depending on used model. Test the setup before final deployment.

The cache may consist of a compiled model blob in a form of `.blob` file or compiled kernels (GPU) in a form of multiple `.cl_cache` files. Cache files can be reused within the same Model Server version, target device, hardware, model and the model shape parameters. 
The Model Server, automatically detects if the cache is present and re-generates new cache files when required. 

Note: Model Server cache feature does not avoid downloading the model files from the remote storage. It speeds up the model loading but access to the original model files is still required.

Note: In some cases model cache might have undesirable side effects. Special considerations are required in the following cases:
- custom loader library is in use - [custom loaders](custom_model_loader.md) might be used to import encrypted model files so using unencrypted cache might potentially lead to a security risk
- using the model with `auto` batch size or `auto` shape parameters - it can lead to model cache regenerating each time the model gets reloaded for new inference requests with new input shape. 
It could potentially cause disk space overloading and it will improve initialization performance only for repeated input shape.


## Enabling the cache functionality

The models caching can be enabled by creating or mounting in the docker container a folder `/opt/cache`. 
Alternatively the location of the cache storage can be set using the parameter `--cache_dir`. 

The model server security context must have read-write access to the cache storage path.

When using Model Server with configuration file, it is possible to serve more than one model. In such case, model cache is applied to all the models, with an exception to:
- Models with custom loader (for security reasons explained earlier)
- Models configured to shape `auto` or batch_size `auto`

In case there are valid reasons to enable the model cache also for models with auto shape or auto batch, it is possible to force enablement with `"allow_cache": true` parameter:
```
{
    "model_config_list": [
        {"config": {
            "name": "face_detection",
            "base_path": "/models/face-detection-retail-0004/",
            "shape": "auto",
            "allow_cache": true}},
    ],
}
```

> IMPORTANT: Models imported via the custom loaders never create or use any cache.

## Use case example

### Prepare model
```bash
$ curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o model/fdsample/1/face-detection-retail-0004.bin -o model/fdsample/1/face-detection-retail-0004.xml
```

### Starting the service

@sphinxdirective
.. code-block:: sh

    $ mkdir cache
    $ docker run -p 9000:9000 -d -u $(id -u):$(id -g) --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) -v ${PWD}/model/fdsample:/model:ro -v ${PWD}/cache:/opt/cache:rw openvino/model_server:latest-gpu --model_name model --model_path /model --target_device GPU --port 9000

@endsphinxdirective

Expected message in the logs `Model cache is enabled: /opt/cache`.

The first time the model server container is started, it will populate the cache folder. The next time the container starts, the initialization will be faster, especially for the GPU target device.

Logs from the first initialization - model loading takes ~3.5s
```
[2021-11-12 16:03:43.325][1][serving][info][modelinstance.cpp:558] Loading model: model, version: 1, from path: /model/1, with target device: GPU ...
[2021-11-12 16:03:43.325][1][serving][info][modelversionstatus.hpp:155] STATUS CHANGE: Version 1 of model model status change. New status: ( "state": "START", "error_code": "OK" )
[2021-11-12 16:03:43.325][1][serving][info][modelversionstatus.hpp:155] STATUS CHANGE: Version 1 of model model status change. New status: ( "state": "LOADING", "error_code": "OK" )
[2021-11-12 16:03:43.342][1][serving][info][modelinstance.cpp:175] Final network inputs:
Input name: data; mapping_name: ; shape: (1,3,300,300); effective shape: (1,3,300,300); precision: FP32; layout: N...
[2021-11-12 16:03:43.342][1][serving][info][modelinstance.cpp:209] Output name: detection_out; mapping name: ; shape: 1 1 200 7 ; effective shape 1 1 200 7 ; precision: FP32; layout: N...
[2021-11-12 16:03:46.905][1][modelmanager][info][modelinstance.cpp:394] Plugin config for device GPU:
[2021-11-12 16:03:46.905][1][modelmanager][info][modelinstance.cpp:398] OVMS set plugin settings key:GPU_THROUGHPUT_STREAMS; value:GPU_THROUGHPUT_AUTO;
[2021-11-12 16:03:46.911][1][serving][info][modelinstance.cpp:477] Loaded model model; version: 1; batch size: 1; No of InferRequests: 4
[2021-11-12 16:03:46.911][1][serving][info][modelversionstatus.hpp:155] STATUS CHANGE: Version 1 of model model status change. New status: ( "state": "AVAILABLE", "error_code": "OK" )
```

Sequential model server initialization is faster. Based on logs below, it is ~400ms.
```
[2021-11-12 16:06:08.377][1][serving][info][modelinstance.cpp:558] Loading model: model, version: 1, from path: /model/1, with target device: GPU ...
[2021-11-12 16:06:08.377][1][serving][info][modelversionstatus.hpp:155] STATUS CHANGE: Version 1 of model model status change. New status: ( "state": "START", "error_code": "OK" )
[2021-11-12 16:06:08.377][1][serving][info][modelversionstatus.hpp:155] STATUS CHANGE: Version 1 of model model status change. New status: ( "state": "LOADING", "error_code": "OK" )
[2021-11-12 16:06:08.384][1][serving][info][modelinstance.cpp:175] Final network inputs:
Input name: data; mapping_name: ; shape: (1,3,300,300); effective shape: (1,3,300,300); precision: FP32; layout: N...
[2021-11-12 16:06:08.384][1][serving][info][modelinstance.cpp:209] Output name: detection_out; mapping name: ; shape: 1 1 200 7 ; effective shape 1 1 200 7 ; precision: FP32; layout: N...
[2021-11-12 16:06:08.783][1][modelmanager][info][modelinstance.cpp:394] Plugin config for device GPU:
[2021-11-12 16:06:08.783][1][modelmanager][info][modelinstance.cpp:398] OVMS set plugin settings key:GPU_THROUGHPUT_STREAMS; value:GPU_THROUGHPUT_AUTO;
[2021-11-12 16:06:08.790][1][serving][info][modelinstance.cpp:477] Loaded model model; version: 1; batch size: 1; No of InferRequests: 4
[2021-11-12 16:06:08.790][1][serving][info][modelversionstatus.hpp:155] STATUS CHANGE: Version 1 of model model status change. New status: ( "state": "AVAILABLE", "error_code": "OK" )
```

