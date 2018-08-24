# Using OpenVINO&trade; model server in a docker container

## Building the image

OpenVINO&trade; model server docker image can be built using the included [Dockerfile](../Dockerfile). It is tested with Ubuntu16.04 as the base image. It should be fairly simple to adjust the process to CentoOS base images.

Before you start building the docker image, you need to download [OpenVINO&trade; toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download) and place the .tgz file in the repository root folder along the Dockerfile. A registration process is required to download the toolkit.
It is recommended to use online installation package because this way the resultant image will be smaller. 
An example file looks like: `l_openvino_toolkit_fpga_p_2018.2.300_online.tgz`.


From the root of the git repository, execute the command:

```bash
docker build -f Dockerfile -t ie-serving-py:latest .
```

If you are building behind the proxy, include also http proxy build parameters:
```bash
docker build -f Dockerfile --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t ie-serving-py:latest .

```

## Preparing the models

After the docker image is built, you can use it to start the model server container, but you should start from preparing the models to be served.

AI models should be created in Intermediate Representation (IR) format (a pair of files with .bin and .xml extensions). 
OpenVINO&trade; toolkit includes a `model_optimizer` tool for converting  TensorFlow, Caffe and MXNet trained models into IR format.  
Refer to the [model optimizer documentation](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) for more details.

Predefined IR models should be placed and mounted in a folder structure as depicted below:
```bash
tree models/
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── ir_model.bin
│       └── ir_model.xml
└── model2
    └── 1
        ├── ir_model.bin
        ├── ir_model.xml
        └── tensors_mapping.json
``` 

Each model should be stored in a dedicated folder (model1 and model2 in the examples above) and should include subfolders
representing its versions. The versions and the subfolder names should be positive integer values. 

Every version folder must include a pair of model files with .bin and .xml extensions while the file name can be arbitrary.

Each model in IR format defines input and output tensors in the AI graph. By default OpenVINO&trade; model server is using 
tensors names as the input and output dictionary keys.  The client is passing the input values to the gRPC request and 
reads the results by referring to the correspondent tensor names. 

Below is the snippet of the example client code:
```python
input_tensorname = 'input'
request.inputs[input_tensorname].CopyFrom(make_tensor_proto(img, shape=(1, 3, 224, 224)))

.....

output_tensorname = 'resnet_v1_50/predictions/Reshape_1'
predictions = make_ndarray(result.outputs[output_tensorname])
```

It is possible to adjust this behavior by adding an optional json file with name `tensors_mapping.json` 
which can map the input and output keys to the appropriate tensors.

```json
{
       "inputs": 
           { "tensor_name":"grpc_custom_input_name"},
       "outputs":{
        "tensor_name1":"grpc_output_key_name1",
        "tensor_name2":"grpc_output_key_name2"
       }
}
```
This extra mapping can be handy to enable model `user friendly` names on the client when the model has cryptic 
tensor names.

OpenVINO&trade; model server is enabling all the versions present in the configured model folder. If you would like to limit 
the versions exposed, for example to reduce the mount of RAM, you need to delete the subfolders representing unnecessary model versions.

While the client is not defining the model version in the request specification, OpenVINO&trade; model server will use the latest one stored in the subfolder of the highest number.


## Starting docker container with a single model

When the models are ready and stored in correct folders structure, you are ready to start the docker container with the 
OpenVINO&trade; model server. To enable just a single model, you don't need any extra configuration file, so this process can be completed with just one command like below:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/model1 --model_name my_model --port 9001
```

* option `-v` defines how the models folder should be mounted inside the docker container.

* option `-p` exposes the model serving port outside the docker container.

* `ie-serving-py:latest` represent the image name which can be different depending the tagging and building process.

* `start_server.sh` script activates the python virtual environment inside the docker container.

* `ie_serving` command starts the model server which has the following parameters:

```bash
usage: ie_serving model [-h] --model_name MODEL_NAME --model_path MODEL_PATH [--port PORT] 

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of the model
  --model_path MODEL_PATH
                        absolute path to model,as in tf serving
  --port PORT           server port

```


If you need to expose multiple models, you need to create a model server configuration file, which is explained in the following section.

## Starting docker container with a configuration file

Model server configuration file defines multiple models, which can be exposed for clients requests.
It uses `json` format as shown in the example below:

```json
{
   "model_config_list":[
      {
         "config":{
            "name":"model_name1",
            "base_path":"/opt/ml/models/model1"
         }
      },
      {
         "config":{
            "name":"model_name2",
            "base_path":"/opt/ml/models/model2"
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"/opt/ml/models/model3"
         }
      }
   ]
}

```
It has a mandatory array `model_config_list`, which includes a collection of `config` objects for each served model. 
Each config object includes values for the model `name` and the `base_path` attributes.

When the config file is present, the docker container can be started in a similar manner as a single model:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001
```

Below is the explanation of the `ie_serving config` parameters
```bash
usage: ie_serving config [-h] --config_path CONFIG_PATH [--port PORT]

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        absolute path to json configuration file
  --port PORT           server port
```

## Batch processing

Inference processing can be executed in batches when the OpenVINO&trade; model are exported by the model optimizer with batch size >1 or the size of first dimension is >1.

Generally OpenVINO&trade; model server determines the batch size based on the size of the first dimension in the first input.
Dynamic batch size is not supported.

For example with the input shape (1, 3, 225, 225), the batch size is set to 1. With input shape (8, 3, 225, 225) the batch size is set to 8.

From the performance point of view, processing a batch of requests is usually more efficient than executing sequentially.
The side effect of increased throughput will be higher latency.
