# ie-serving-py
Inference serving implementation with gRPC interface and DLDT in the backend


## Project overview


## Building
If you want work in virtualenv please provide command in root directory of this repository:
```
make install
```

If you want to contribute to this repository and make changes in code the easiest way to install this package is provide command: 
```
pip install -e .
```

## Service configuration
Before start service you have to install Intel OpenVino. Refer to Dockerfile as a reference how it can be done 
or follow the documentation from https://software.intel.com/en-us/openvino-toolkit


## Setting CPU extention library

You need you change the default value of CPU_EXTENSION library path in the following situations:
* you install Intel OpenVino in non default path which is /opt/intel/computer_vision_sdk
* your HW does not support AVX2 cpu feature 
* you use non ubuntu OS to host is-serving-py service

The default value of CPU_EXTENSION is:
```
/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so
```

## Starting inference service

To start server with one model:
```
ie_serving model --model_path <absolute_path_to_model> --model_name <model_name>
```

To start server with multiple models please prepare config like this:
```json
{
   "model_config_list":[
      {
         "config":{
            "name":"resnet1",
            "base_path":"/home/marek/models/grpc_model"
         }
      },
      {
         "config":{
            "name":"resnet2",
            "base_path":"/home/marek/models/grpc_model"
         }
      },
      {
         "config":{
            "name":"resnet3",
            "base_path":"/home/marek/models/grpc_model"
         }
      }
   ]
}
```
Next use command:
```
ie_serving config --config_path <absolute_path_to_config>
```

Optionally you can specify port(default=9000) and maximum workers for the server(default=10)
```
ie_serving model --model_path <absolute_path_to_model> --model_name <model_name> --port 9999 --max_workers 1
```
or for config
```
ie_serving config --config_path <absolute_path_to_config> --port 9999
```

### Starting ie-sevice-py

Follow example from `make docker_run`


## Testing

### Python style tests
`make style`
It executes style verification using flake8


### Functional tests
Functional testing can be triggered via:

`make test`

Tests should be preceded by building a docker image with `make docker_build`. 
During tests there will be downloaded sample models along with sample datasets to 
execute inference against local docker container 


## Known limitations

For now only Predict calls are implemented using Tensorflow Serving API.
