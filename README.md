# ie-serving-py
Inference serving implementation with gRPC interface and DLDT in the backend


## Project overview


## Building
If you want work in virtualenv please provide command:
```
make install
```
In root directory of this repository please provide command:
```
pip install .
```

If you want to contribute to this repository and make changes in code the easiest way to install this package is provide command: 
```
pip install -e .
```

## Service configuration
Before start service you have to install intel OpenVino

And provide this command(this is an default path to this script):
```
source /opt/intel/deeplearning_deploymenttoolkit/deployment_tools/inference_engine/bin/setvars.sh
```
## Starting inference service
You have to set env file which will be specify to CPU_EXTENSION. In case of my OpenVino installation this path is:
```
/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so
```
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
## Testing


## Known limitations

