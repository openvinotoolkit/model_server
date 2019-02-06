# Using OpenVINO&trade; Model Server in a Docker Container

## Building the Image

OpenVINO&trade; model server Docker image can be built from [DLDT sources](https://github.com/opencv/dldt) with 
ubuntu base image [Dockerfile](../Dockerfile), intelpython base image [Dockerfile_intelpython](../Dockerfile_intelpython)
or with Intel Distribution of OpenVINO&trade; [toolkit package](https://software.intel.com/en-us/openvino-toolkit)
via [Dockerfile_binary_openvino](../Dockerfile_binary_openvino).

The latter option requires downloaded [OpenVINO&trade; toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download) and placed in the repository root folder along the Dockerfile. A registration process is required to download the toolkit.
It is recommended to use online installation package because this way the resultant image will be smaller. 
An example file looks like: `l_openvino_toolkit_p_2018.5.445_online.tgz`.


From the root of the git repository, execute the command:

```bash
cp (download path)/l_openvino_toolkit_p_2018.5.445_online.tgz . 
make docker_build_bin http_proxy=$http_proxy https_proxy=$https_proxy
```
or
```bash
make docker_build_src_ubuntu http_proxy=$http_proxy https_proxy=$https_proxy
```
or
```bash
make docker_build_src_intelpython http_proxy=$http_proxy https_proxy=$https_proxy
```

**Note:** You can use also publicly available docker image from [dockerhub](https://hub.docker.com/r/intelaipg/openvino-model-server/)

```bash
docker pull intelaipg/openvino-model-server
```

## Preparing the Models

After the Docker image is built, you can use it to start the model server container, but you should start from preparing the models to be served.

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
        └── mapping_config.json
``` 

Each model should be stored in a dedicated folder (model1 and model2 in the examples above) and should include subfolders
representing its versions. The versions and the subfolder names should be positive integer values. 

Every version folder _must_ include a pair of model files with .bin and .xml extensions; however, the file name can be arbitrary.

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

It is possible to adjust this behavior by adding an optional json file with name `mapping_config.json` 
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

OpenVINO&trade; model server is enabling all the versions present in the configured model folder. To limit 
the versions exposed, for example to reduce the mount of RAM, you need to delete the subfolders representing unnecessary model versions.

While the client _is not_ defining the model version in the request specification, OpenVINO&trade; model server will use the latest one stored in the subfolder of the highest number.


## Starting Docker Container with a Single Model

When the models are ready and stored in correct folders structure, you are ready to start the Docker container with the 
OpenVINO&trade; model server. To enable just a single model, you _do not_ need any extra configuration file, so this process can be completed with just one command like below:

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
usage: ie_serving model [-h] --model_name MODEL_NAME --model_path MODEL_PATH
                        [--batch_size BATCH_SIZE] [--port PORT]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of the model
  --model_path MODEL_PATH
                        absolute path to model,as in tf serving
  --batch_size BATCH_SIZE
                        sets models batchsize, int value or auto
  --port PORT           server port

```


The model path could be local on docker container like mounted during startup or it could be Google Cloud Storage path 
in a format `gs://<bucket>/<model_path>`. In this case it will be required to 
pass GCS credentials to the docker container,
unless GKE kubernetes cluster, which handled the authorization automatically,
 is used.

Below is an example presenting how to start docker container with a support for GCS paths to the models. The variable 
`GOOGLE_APPLICATION_CREDENTIALS` contain a path to GCP authentication key. 

```bash
docker run --rm -d  -p 9001:9001 ie-serving-py:latest \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}”  \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}
/ie-serving-py/start_server.sh ie_serving model --model_path gs://bucket/model_path --model_name my_model --port 9001
```

Learn [more about GCP authentication](https://cloud.google.com/docs/authentication/production).


It is also possible to provide paths to models located in S3 compatible storage
in a format `s3://<bucket>/<model_path>`. In this case it is necessary to 
provide credentials to bucket by setting environmental variables
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. You can also set 
`AWS_REGION` variable, although it's not always required. 
If you are using custom storage server compatible with S3, you must set `S3_ENDPOINT` 
environmental variable in a HOST:PORT format. In an example below you can see 
how to start docker container serving single model located in S3.

```bash
docker run --rm -d  -p 9001:9001 ie-serving-py:latest \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}”  \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}”  \
-e AWS_REGION=“${AWS_REGION}”  \
-e S3_ENDPOINT=“${S3_ENDPOINT}”  \
/ie-serving-py/start_server.sh ie_serving model --model_path 
s3://bucket/model_path --model_name my_model --port 9001 --batch_size auto
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
            "base_path":"/opt/ml/models/model1",
            "batch_size": "16"
         }
      },
      {
         "config":{
            "name":"model_name2",
            "base_path":"/opt/ml/models/model2",
            "batch_size": "auto"
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3"
         }
      },
      {
         "config":{
             "name":"model_name4",
             "base_path":"s3://bucket/models/model4"
         }
      }
   ]
}

```
It has a mandatory array `model_config_list`, which includes a collection of `config` objects for each served model. 
Each config object includes values for the model `name` and the `base_path` attributes.

When the config file is present, the docker container can be started in a 
similar manner as a single model. Keep in mind that models with cloud 
storage path require specific environmental variables set. Configuration 
file above contains both GCS and S3 paths so starting docker container 
supporting all those models can be done with:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 ie-serving-py:latest \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}”  \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}  \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}”  \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}”  \
-e AWS_REGION=“${AWS_REGION}”  \
-e S3_ENDPOINT=“${S3_ENDPOINT}”  \
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

## Batch Processing

`batch_size` parameter is optional. By default is accepted the batch size derived from the model. It is set by the model optimizer.
When that parameter is set to numerical value, it is changing the model batch size at service start up. 
It accepts also a value `auto` - this special phrase make the served model to set the batch size automatically based on the incoming data at run time.
Each time the input data change the batch size, the model is reloaded. It might have extra response delay for the first request.
This feature is useful for sequential inference requests of the same batch size.

OpenVINO&trade; Model Server determines the batch size based on the size of the first dimension in the first input.
For example with the input shape (1, 3, 225, 225), the batch size is set to 1. With input shape (8, 3, 225, 225) the batch size is set to 8.

**Note:** Dynamic batch size _is not_ supported.

Processing bigger batches of requests increases the throughput but the side effect is higher latency.
