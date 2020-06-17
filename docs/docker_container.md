# Using OpenVINO&trade; Model Server in a Docker Container

## Building the Image

Build the docker image using command:
 
```bash
~/ovms-c$ make docker_build BASE_OS=[one of ubuntu/centos/clearlinux ]
```
It will generate the image, tagged as `ovms:latest`, as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.

The release package should work on a any linux machine with glibc >= one used by the build image.

For debugging, an image with a suffix `-build` is also generated (i.e. `ovms-build:latest`).


**Note:** You can use also publicly available docker image from internal docker registry service.


```bash
docker pull ger-registry-pre.caas.intel.com/ovms/model_server:latest
docker tag ger-registry-pre.caas.intel.com/ovms/model_server:latest ovms:latest
```

Before deploying OVMS server [prepare models and models repository](models_repository.md).


## Starting Docker Container with a Single Model

When the models are ready and stored in correct folders structure, you are ready to start the Docker container with the 
OpenVINO&trade; model server. To enable just a single model, you _do not_ need any extra configuration file, so this process can be 
completed with just one command like below:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ovms:latest \
--model_path /opt/ml/model1 --model_name my_model --port 9001 --rest_port 8001
```

* option `-v` defines how the models folder should be mounted inside the docker container.

* option `-p` exposes the model serving port outside the docker container.

* `ovms:latest` represent the image name which can be different depending the tagging and building process.

* `ovms` binary is the docker entrypoint. It accepts the following parameters:

```bash
OpenVINO Model Server
Usage:
  /ovms/bin/ovms [OPTION...]

  -h, --help                    show this help message and exit
      --port PORT               gRPC server port (default: 9178)
      --rest_port REST_PORT     REST server port, the REST server will not be
                                started if rest_port is blank or set to 0
                                (default: 0)
      --grpc_workers GRPC_WORKERS
                                number of gRPC servers. Recommended to be >=
                                NIREQ. Default value calculated at runtime:
                                NIREQ + 2 
      --rest_workers REST_WORKERS
                                number of workers in REST server - has no
                                effect if rest_port is not set 
      --log_level LOG_LEVEL     serving log level - one of DEBUG, INFO, ERROR
                                (default: INFO)
      --log_path LOG_PATH       optional path to the log file
      --grpc_channel_arguments GRPC_CHANNEL_ARGUMENTS
                                A comma separated list of arguments to be
                                passed to the grpc server. (e.g.
                                grpc.max_connection_age_ms=2000)

 multi model options:
      --config_path CONFIG_PATH
                                absolute path to json configuration file

 single model options:
      --model_name MODEL_NAME   name of the model
      --model_path MODEL_PATH   absolute path to model, as in tf serving
      --batch_size BATCH_SIZE   sets models batchsize, int value or auto.
                                This parameter will be ignored if shape is set
                                (default: 0)
      --shape SHAPE             sets models shape (model must support
                                reshaping). If set, batch_size parameter is ignored
      --model_version_policy MODEL_VERSION_POLICY
                                model version policy
      --nireq NIREQ             Number of parallel inference request
                                executions for model. Recommended to be >=
                                CPU_THROUGHPUT_STREAMS. Default value calculated at
                                runtime: CPU cores / 8
      --target_device TARGET_DEVICE
                                Target device to run the inference (default:
                                CPU)
      --plugin_config PLUGIN_CONFIG
                                a dictionary of plugin configuration keys and
                                their values, eg
                                "{\"CPU_THROUGHPUT_STREAMS\": \"CPU_THROUGHPUT_AUTO\"}"
```


The model path could be local on docker container like mounted during startup or it could be Google Cloud Storage path 
in a format `gs://<bucket>/<model_path>`. In this case it will be required to 
pass GCS credentials to the docker container,
unless GKE kubernetes cluster, which handled the authorization automatically,
 is used.

Below is an example presenting how to start docker container with a support for GCS paths to the models. The variable 
`GOOGLE_APPLICATION_CREDENTIALS` contain a path to GCP authentication key. 

```bash
docker run --rm -d  -p 9001:9001 ovms:latest \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}”  \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}
--model_path gs://bucket/model_path --model_name my_model --port 9001
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
docker run --rm -d  -p 9001:9001 ovms:latest \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}”  \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}”  \
-e AWS_REGION=“${AWS_REGION}”  \
-e S3_ENDPOINT=“${S3_ENDPOINT}”  \
--model_path s3://bucket/model_path --model_name my_model --port 9001 --batch_size auto --model_version_policy '{"all": {}}'
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
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 ovms:latest \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}”  \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}  \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}”  \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}”  \
-e AWS_REGION=“${AWS_REGION}”  \
-e S3_ENDPOINT=“${S3_ENDPOINT}”  \
--config_path /opt/ml/config.json --port 9001 --rest_port 8001
```



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



## Starting docker container with NCS

Plugin for [Intel® Movidius™ Neural Compute Stick](https://software.intel.com/en-us/neural-compute-stick), starting from 
version 2019 R1.1 is distributed both in a binary package and [source code](https://github.com/opencv/dldt). 
You can build the docker image of OpenVINO Model Server, including Myriad plugin, using any form of the OpenVINO toolkit distribution:
- `make docker_build ` 

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
ovms:latest --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

`--net=host` and `--privileged` parameters are required for USB connection to work properly. 

`-v /dev:/dev` mounts USB drives.

A single stick can handle one model at a time. If there are multiple sticks plugged in, OpenVINO Toolkit 
chooses to which one the model is loaded. 

## User GPU for Inference execution

TBD

## Starting docker container with HDDL

TBD

## Using Multi-Device Plugin

TBD