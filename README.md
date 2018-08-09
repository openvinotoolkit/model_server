# ie-serving-py
Inference serving implementation with gRPC interface compatible with TensorFlow sering API and OpenVino in the 
execution backend


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
            "name":"model_name1",
            "base_path":"/opt/ml/models/IR_model1"
         }
      },
      {
         "config":{
            "name":"model_name2",
            "base_path":"/opt/ml/models/IR_model2"
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"/opt/ml/models/IR_model3"
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

### Config map for model
Optionally gRPC input and output key names can be mapped to the graph tensor names. This way from gRPC interface individual 
input and outputs can be referenced with different naming convention.
If tensors_mapping.json file is not included in the model version folder along with .bin and .xml model files there will
be set default input and output key names equal to the corespondent tensor names.


tensors_mapping.json
```
{
       "inputs": 
           { "tensor_name":"custom_name"},
       "outputs":{
        "tesnor_name1":"grpc_output_key_name1",
        "tensor_name2":"grpc_output_key_name2"
       }
}
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
