# Model Cache

## Overview
Model cache feature in Model Server leverages [OpenVINO&trade; functionality](https://docs.openvino.ai/latest/openvino_docs_IE_DG_Model_caching_overview.html) to cache network loaded into device to disk during initial loading. Subsequent model loadings are loaded from cache what usually makes it faster. The boost depends on model and target device. The cache may consist of entire network (CPU) in a form of `.blob` file or compiled kernels (GPU) in a form of multiple `.cl_cache` files. Cached files can be reused within the same Model Server version, target device and hardware. Model Server, using OpenVINO&trade; automatically detects if the cache is missing and re-generates new cache files when required.

## Usage
### Prepare model
```
$ curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/fdsample/1/face-detection-retail-0004.xml -o model/fdsample/1/face-detection-retail-0004.bin
```

### Starting the service
By default, model cache feature is turned off. To enable it, mount host directory to `/opt/cache` directory with read and write access when starting container. Keep it mind the host directory must have correct write rights to allow container to write.
```
$ mkdir cache
$ chmod 777 cache
$ docker run -p 9000:9000 -d -v ${PWD}/model/fdsample:/model -v ${PWD}/cache:/opt/cache:rw openvino/model_server --model_name model --model_path /model/ --port 9000
```
Expected logs to appear:
```
[2021-11-03 15:37:28.823][849][modelmanager][info][modelmanager.cpp:85] Model cache is enabled: /opt/cache
```

Once model is loaded, the cache file should appear in directory:
```
$ ls ./cache
258384481600631188.blob
```

## Change default cache directory
When running Model Server on bare metal, it is useful to change the cache directory to other location than `/opt/cache`. To change cache directory use `--cache_dir` command line parameter:
```
$ ./ovms --model_name model --model_path /path/to/model --cache_dir /home/my_user/cache 
```

## Disable cache for selected models
When using Model Server with configuration file, it is possible to serve more than one model. In such case, model cache is applied to all the models, with an exception to:
- Models with custom loader for security reasons since custom loaders can be used to encrypt the model. Dumping the network to cache does not encrypt the network.
- Models configured to shape and batch size `auto`. Shape `auto` makes the model reload if the input shape does not match network shape. Every reshape to new shape is treated by OpenVINO&trade; as different network and new cache files are generated. Since the model cache files are usually large, it might be possible flood the disk using multiple requests with different input shape.

In case the security is not a concern, it is possible to force enablement of model cache for such models with `"force_caching": true` parameter:
```
{
    "model_config_list": [
        {"config": {
            "name": "face_detection",
            "base_path": "/models/face-detection-retail-0004/",
            "shape": "auto",
            "force_caching": true}},
    ],
}
```
