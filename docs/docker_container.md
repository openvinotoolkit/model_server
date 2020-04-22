# Using OpenVINO&trade; Model Server in a Docker Container

## Building the Image

OpenVINO&trade; Model Server docker image can be built using various Dockerfiles:
- [Dockerfile](../Dockerfile) - based on ubuntu with [apt-get package](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_apt.html) 
- [Dockerfile_clearlinux](../Dockerfile_clearlinux) - [clearlinux](https://clearlinux.org/) based image with [DLDT package](https://github.com/clearlinux-pkgs/dldt) included
- [Dockerfile_binary_openvino](../Dockerfile_binary_openvino) - ubuntu image based on Intel Distribution of OpenVINO&trade; [toolkit package](https://software.intel.com/en-us/openvino-toolkit)

The last option requires URL to OpenVINO Toolkit package that you can get after registration on [OpenVINO&trade; Toolkit website](https://software.intel.com/en-us/openvino-toolkit/choose-download).
Use this URL as an argument in the `make` command as shown in example below:

```bash
make docker_build_bin dldt_package_url=<url-to-openvino-package-after-registration>/l_openvino_toolkit_p_2020.1.023_online.tgz
```
or
```bash
make docker_build_apt_ubuntu
```
or
```bash
make docker_build_ov_base
```
or
```bash
make docker_build_clearlinux
```

**Note:** You can use also publicly available docker image from [dockerhub](https://hub.docker.com/r/intelaipg/openvino-model-server/)
based on clearlinux base image.

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

While the client _is not_ defining the model version in the request specification, OpenVINO&trade; model server will use the latest one 
stored in the subfolder of the highest number.


## Starting Docker Container with a Single Model

When the models are ready and stored in correct folders structure, you are ready to start the Docker container with the 
OpenVINO&trade; model server. To enable just a single model, you _do not_ need any extra configuration file, so this process can be 
completed with just one command like below:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/model1 --model_name my_model --port 9001 --rest_port 8001
```

* option `-v` defines how the models folder should be mounted inside the docker container.

* option `-p` exposes the model serving port outside the docker container.

* `ie-serving-py:latest` represent the image name which can be different depending the tagging and building process.

* `start_server.sh` script activates the python virtual environment inside the docker container.

* `ie_serving` command starts the model server which has the following parameters:

```bash
usage: ie_serving model [-h] --model_name MODEL_NAME --model_path MODEL_PATH
                        [--batch_size BATCH_SIZE] [--shape SHAPE]
                        [--port PORT] [--rest_port REST_PORT]
                        [--model_version_policy MODEL_VERSION_POLICY]
                        [--grpc_workers GRPC_WORKERS]
                        [--rest_workers REST_WORKERS] [--nireq NIREQ]
                        [--target_device TARGET_DEVICE]
                        [--plugin_config PLUGIN_CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        name of the model
  --model_path MODEL_PATH
                        absolute path to model,as in tf serving
  --batch_size BATCH_SIZE
                        sets models batchsize, int value or auto. This
                        parameter will be ignored if shape is set
  --shape SHAPE         sets models shape (model must support reshaping). If
                        set, batch_size parameter is ignored.
  --port PORT           gRPC server port
  --rest_port REST_PORT
                        REST server port, the REST server will not be started
                        if rest_port is blank or set to 0
  --model_version_policy MODEL_VERSION_POLICY
                        model version policy
  --grpc_workers GRPC_WORKERS
                        Number of workers in gRPC server. Default: 1
  --rest_workers REST_WORKERS
                        Number of workers in REST server - has no effect if
                        rest port not set. Default: 1
  --nireq NIREQ         Number of parallel inference requests for model. Default: 1
  --target_device TARGET_DEVICE
                        Target device to run the inference, default: CPU
  --plugin_config PLUGIN_CONFIG
                        A dictionary of plugin configuration keys and their values

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
s3://bucket/model_path --model_name my_model --port 9001 --batch_size auto --model_version_policy '{"all": {}}'
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
            "batch_size": "auto",
            "model_version_policy": {"all": {}}
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
            "model_version_policy": {"specific": { "versions":[1, 3] }},
            "shape": "auto"
         }
      },
      {
         "config":{
             "name":"model_name4",
             "base_path":"s3://bucket/models/model4",
             "shape": {
                "input1": "(1,3,200,200)",
                "input2": "(1,3,50,50)"
             },
             "plugin_config": {"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}
         }
      },
      {
         "config":{
             "name":"model_name5",
             "base_path":"s3://bucket/models/model5",
             "shape": "auto",
             "nireq": 32,
             "target_device": "HDDL",
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
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ie-serving-py:latest \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}”  \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}  \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}”  \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}”  \
-e AWS_REGION=“${AWS_REGION}”  \
-e S3_ENDPOINT=“${S3_ENDPOINT}”  \
/ie-serving-py/start_server.sh ie_serving config --config_path /opt/ml/config.json --port 9001 --rest_port 8001
```

Below is the explanation of the `ie_serving config` parameters
```bash
usage: ie_serving config [-h] --config_path CONFIG_PATH [--port PORT]
                         [--rest_port REST_PORT] [--grpc_workers GRPC_WORKERS]
                         [--rest_workers REST_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        absolute path to json configuration file
  --port PORT           gRPC server port
  --rest_port REST_PORT
                        REST server port, the REST server will not be started
                        if rest_port is blank or set to 0
  --grpc_workers GRPC_WORKERS
                        Number of workers in gRPC server
  --rest_workers REST_WORKERS
                        Number of workers in REST server - has no effect if
                        rest_port is not set

```

## Starting docker container with NCS

Plugin for [Intel® Movidius™ Neural Compute Stick](https://software.intel.com/en-us/neural-compute-stick), starting from 
version 2019 R1.1 is distributed both in a binary package and [source code](https://github.com/opencv/dldt). 
You can build the docker image of OpenVINO Model Server, including Myriad plugin, using any form of the OpenVINO toolkit distribution:
- `make docker_build_bin dldt_package_url=<url>` 
- `make docker_build_apt_ubuntu`

Neural Compute Stick must be visible and accessible on host machine. You may need to update udev 
rules:
<details>
<summary><i>Updating udev rules</i></summary>
</br>

1. Create file __97-usbboot.rules__ and fill it with:

```
   SUBSYSTEM=="usb", ATTRS{idProduct}=="2150", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1" 
   SUBSYSTEM=="usb", ATTRS{idProduct}=="2485", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
   SUBSYSTEM=="usb", ATTRS{idProduct}=="f63b", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
```   
2. In the same directory execute following: 
 ```
   sudo cp 97-usbboot.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   sudo ldconfig
   rm 97-usbboot.rules
```

</details>
</br>

To start server with NCS you can use command similar to:

```
docker run --rm -it --net=host -u root --privileged -v /opt/model:/opt/model -v /dev:/dev -p 9001:9001 \
ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

`--net=host` and `--privileged` parameters are required for USB connection to work properly. 

`-v /dev:/dev` mounts USB drives.

A single stick can handle one model at a time. If there are multiple sticks plugged in, OpenVINO Toolkit 
chooses to which one the model is loaded. 

## Starting docker container with HDDL

Plugin for High-Density Deep Learning (HDDL) accelerators based on [Intel Movidius Myriad VPUs](https://www.intel.ai/intel-movidius-myriad-vpus/#gs.xrw7cj).
is distributed only in a binary package. You can build the docker image of OpenVINO Model Server, including HDDL plugin
, using OpenVINO toolkit binary distribution:
- `make docker_build_bin dldt_package_url=<url>` 

In order to run container that is using HDDL accelerator, _hddldaemon_ must
 run on host machine. It's  required to set up environment 
 (the OpenVINO package must be pre-installed) and start _hddldaemon_ on the
  host before starting a container. Refer to the steps from [OpenVINO documentation](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_docker_linux.html#build_docker_image_for_intel_vision_accelerator_design_with_intel_movidius_vpus).

To start server with HDDL you can use command similar to:

```
docker run --rm -it --device=/dev/ion:/dev/ion -v /var/tmp:/var/tmp -v /opt/model:/opt/model -p 9001:9001 \
ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model --model_path /opt/model --model_name my_model --port 9001 --target_device HDDL
```

`--device=/dev/ion:/dev/ion` mounts the accelerator.

`-v /var/tmp:/var/tmp` enables communication with _hddldaemon_ running on the
 host machine

Check out our recommendations for [throughput optimization on HDDL](performance_tuning.md#hddl-accelerators)

## Batch Processing

`batch_size` parameter is optional. By default is accepted the batch size derived from the model. It is set by the model optimizer tool.
When that parameter is set to numerical value, it is changing the model batch size at service start up. 
It accepts also a value `auto` - this special phrase make the served model to set the batch size automatically based on the incoming data at run time.
Each time the input data change the batch size, the model is reloaded. It might have extra response delay for the first request.
This feature is useful for sequential inference requests of the same batch size.

OpenVINO&trade; Model Server determines the batch size based on the size of the first dimension in the first input.
For example with the input shape (1, 3, 225, 225), the batch size is set to 1. With input shape (8, 3, 225, 225) the batch size is set to 8.

*Note:* Some models like object detection do not work correctly with batch size changed with `batch_size` parameter. Typically those are the models,
whose output's first dimension is not representing the batch size like on the input side.
Changing batch size in this kind of models can be done with network reshaping by setting `shape` parameter appropriately.

## Model reshaping
`shape` parameter is optional and it takes precedence over batch_size parameter. When the shape is defined as an argument,
it ignores the batch_size value.

The shape argument can change the model enabled in the model server to fit the required parameters. It accepts 3 forms of the values:
- "auto" phrase - model server will be reloading the model with the shape matching the input data matrix. 
- a tuple e.g. (1,3,224,224) - it defines the shape to be used for all incoming requests for models with a single input
- a dictionary of tuples e.g. {input1:(1,3,224,224),input2:(1,3,50,50)} - it defines a shape of every included input in the model

*Note:* Some models do not support reshape operation. Learn more about supported model graph layers including all limitations
on [docs_IE_DG_ShapeInference.html](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_ShapeInference.html).
In case the model can't be reshaped, it will remain in the original parameters and all requests with incompatible input format
will get an error. The model server will also report such problem in the logs.

## Model Version Policy
Model version policy makes it possible to decide which versions of model will be served by OVMS. This parameter allows 
you to control the memory consumption of the server and 
decide which versions will be used regardless of what is located under the path given when the server is started.
`model_version_policy` parameter is optional. By default server serves only latest version for model. 
Accepted format for parameter in CLI and in config is `json`.
Accepted values:
```
{"all": {}}
{"latest": { "num_versions": Integer}
{"specific": { "versions": List }}
```
Examples:
```
{"latest": { "num_versions":2 } # server will serve only 2 latest versions of model
{"specific": { "versions":[1, 3] }} # server will serve only 1 and 3 versions of given model
{"all": {}} # server will serve all available versions of given model
```

## Updating model versions

Served versions are updated online by monitoring file system changes in the model storage. OpenVINO Model Server
will add new version to the serving list when new numerical subfolder with the model files is added. The default served version
will be switched to the one with the highest number.
When the model version is deleted from the file system, it will become unavailable on the server and it will release RAM allocation.
Updates in the model version files will not be detected and they will not trigger changes in serving.

By default model server is detecting new and deleted versions in 1 second intervals. 
The frequency can be changed by setting environment variable `FILE_SYSTEM_POLL_WAIT_SECONDS`.
If set to negative or zero, updates will be disabled.

## Using Multi-Device Plugin

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. For more detailed information about OpenVino's Multi-Device plugin, see: https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_MULTI.html

In order to use this feature in OpenVino&trade; Model Server, following steps are required:
* Set `target_device` for the model in configuration json file to `MULTI:<DEVICE_1>,<DEVICE_2>` (e.g. `MULTI:MYRIAD,CPU`, order of the devices defines their priority, so `MYRIAD` devices will be used first in this example)
* Set `nireq` (number of inference requests) for the model in configuration json file to be at least equal to the number of the devices that will be used - optimal number of inference requests may vary depending on the model and type of used devices. Following script will help you to get optimal number of requests suggested by OpenVino&trade;:
```python
from openvino.inference_engine import IEPlugin, IENetwork

# replace MULTI:MYRIAD,CPU if you are using different multi device configuration
plugin = IEPlugin(device='MULTI:MYRIAD,CPU')
# change paths to location of your model
model_xml = "<model's xml file path>"
model_bin = "<model's bin file path>"
net = IENetwork(model=model_xml, weights=model_bin)
exec_net = plugin.load(network=net)
print(exec_net.get_metric('OPTIMAL_NUMBER_OF_INFER_REQUESTS'))
```
* Set `grpc_workers` (or `rest_workers` if you are using REST endpoints for inference) parameter to be at least equal to the number of inference requests

### Multi-Device Plugin configuration example

Example of setting up Multi-Device Plugin for resnet model, using Intel® Movidius™ Neural Compute Stick and CPU devices:

Content of `config.json`:
```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "nireq": 6,
      "target_device": "MULTI:MYRIAD,CPU"}
   }]
}
```

Starting OpenVINO&trade; Model Server with `config.json` (placed in `./models/config.json` path) defined as above, and with `grpc_workers` parameter set to match `nireq` field in `config.json`:
```bash
docker run -d  --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro \
-v /dev:/dev -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh \
ie_serving config --config_path /opt/ml/config.json --port 9001 --grpc_workers 6
```

Or alternatively, when you are using just a single model, start OpenVINO&trade; Model Server using this command (`config.json` is not needed in this case):
```
docker run -d  --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro -v \
 /dev:/dev -p 9001:9001 ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving \
 model --model_path /opt/ml/resnet --model_name resnet --port 9001 --grpc_workers 6 \
 --nireq 6 --target_device 'MULTI:MYRIAD,CPU'
```

After these steps, deployed model will perform inference on both Intel® Movidius™ Neural Compute Stick and CPU, and total throughput will be roughly equal to sum of CPU and Intel® Movidius™ Neural Compute Stick throughput.