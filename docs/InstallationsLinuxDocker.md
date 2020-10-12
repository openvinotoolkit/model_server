# Installing OpenVINO&trade; Model Server for Linux using Docker Container

## Introduction

 OpenVINO&trade; Model Server is a serving system for machine learning models.  OpenVINO&trade; Model Server  makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. This guide will help you install  OpenVINO&trade; Model Server for Linux through docker containers.

## System Requirements

### Hardware 
* 6th to 10th generation Intel® Core™ processors and Intel® Xeon® processors
* Intel® Neural Compute Stick 2
* Intel® Vision Accelerator Design with Intel® Movidius™ VPUs

### Operating Systems

* Ubuntu 18.04.x long-term support (LTS), 64-bit

### Overview 

This guide provides step-by-step instructions on how to install OpenVINO&trade; Model Server for Linux using Docker Container including a Quick Start guide. Links are provided for different compatible hardwares. Following instructions are covered in this:

- <a href="#ExistingDocker">Installing OpenVINO&trade; Model Server with existing Docker Container</a>
- <a href="#quickstart">Quick Start Guide for OpenVINO&trade; Model Server</a>
- <a href="#sourcecode">Building the OpenVINO&trade; Model Server Image using Source Code </a>
- <a href="#singlemodel">Starting Docker Container with a Single Model
- <a href="#configfile">Starting Docker container with a configuration file</a>
- <a href="#ncs">Starting Docker container with Neural Compute Stick</a>
- <a href="#hddl">Starting Docker container with HDDL Plugin </a>
- <a href="#gpu">Starting Docker container with GPU</a>
- <a href="#multiplugin">Starting Docker Container using Multi-Device Plugin</a>
- <a href="#heteroplugin">Starting Docker Container using Heterogenous Plugin </a>



## 1. Installing OpenVINO&trade; Model Server with existing Docker Container<a name="ExistingDocker"></a>

A quick start guide to install model server and run it with face detection model is provided below. It includes scripts to query the gRPC endpoints and save results.

For additional endpoints, refer the [REST API](./ModelServerRESTAPI.md)

### Quick Start Guide <a name="quickstart"></a>

```bash
 # Pull the latest version of OpenVINO&trade; Model Server from Dockerhub - 
docker pull openvino/model_server:latest

#  Download model files into a separate directory
# OpenVINO&trade; Model Server requires models in model repositories. Refer to this link (Preparation of Models) or run following command to get started with an example

curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/face-detection-retail-0004.xml -o model/face-detection-retail-0004.bin

#  Start the container serving gRPC on port 9000 
docker run -d -v <folder_with_downloaded_model>:/models/face-detection/1 -p 9000:9000 openvino/model_server:latest \
--model_path /models/face-detection --model_name face-detection --port 9000 --log_level DEBUG --shape auto

#  Download the example client script to test and run the gRPC 
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/face_detection.py -o face_detection.py  https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/client_requirements.txt -o client_requirements.txt

# Download an image to be analyzed
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/images/people/people1.jpeg -o images/people1.jpeg

# Install python client dependencies
pip install -r client_requirements.txt

#  Create a directory for results
mkdir results

# Run inference and store results in the newly created folder
python face_detection.py --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results

# Check results folder for image with inference drawn over it.
```


### Detailed steps to install OpenVINO&trade; Model Server using Docker container

#### Install Docker

Install Docker for Ubuntu 18.04 using the following link

- [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

#### Pulling OpenVINO&trade; Model Server Image

After Docker installation you can pull the OpenVINO&trade; Model Server image. Open Terminal and run following command:

```bash
docker pull openvino/model_server:latest

```
You can also build your own image using steps in - <a href="#sourcecode">Build OpenVINO&trade; Model Server Image </a>

#### Running the OpenVINO&trade; Model Server image

Follow the [Preparation of Model guide](./PreparingModelsRepository.md) before running the docker image 

Run the OpenVINO&trade; Model Server by running the following command: 

```
docker run -d -v <folder_with_downloaded_model>:/models/face-detection/1 -e LOG_LEVEL=DEBUG -p 9000:9000 -p 9001:9001 openvino/model_server:latest \
--model_path path_to_model --model_name model_name --port 9000 --rest_port 9001 --shape auto
```

- Publish the container's port to your host's **open ports**
- In above command port 9000 is exposed for gRPC and port 9001 is exposed for REST API calls.
- For preparing and saving models to serve with OpenVINO&trade; Model Server refer [this](./PreparingModelsRepository.md)
- Add a name to your model for the client gRPC/REST API calls.

#### Other Arguments

Additional arguments that can be used while running the docker image:

```
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
      --file_system_poll_wait_seconds SECONDS
                                Time interval between config and model versions 
                                changes detection. Default is 1. Zero or negative
                                value disables changes monitoring.

 multi model options:
      --config_path CONFIG_PATH
                                absolute path to json configuration file

 single model options:
      --model_name MODEL_NAME   name of the model
      --model_path MODEL_PATH   absolute path to model, as in tf serving
      --batch_size BATCH_SIZE   sets models batch size, int value or auto.
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
                                their values, eg "{\"CPU_THROUGHPUT_STREAMS\": \"1\"}".
                                Default throughput streams for CPU and GPU are calculated by OpenVINO

               
```

#### To know more about batch size and shape parameters refer [Batch Size and Shape document](./ShapeAndBatchSize.md)

More details about starting container with one model and examples can be found <a href="#singlemodel">here</a>

## 2. Building the OpenVINO&trade; Model Server Image<a name="sourcecode"></a>

To build your own image, use the following command in the [git repository root folder](https://github.com/openvinotoolkit/model_server), replacing DLDT_PACKAGE_URL=<URL> with the URL to OpenVINO Toolkit package that you can get after registration on OpenVINO™ Toolkit website.

```bash
   make docker_build DLDT_PACKAGE_URL=<URL>
```
called from the root directory of the repository.

It will generate the images, tagged as:

- openvino/model_server:latest - with CPU, NCS and HDDL support
- openvino/model_server-gpu:latest - with CPU, NCS, HDDL and iGPU support
as well as a release package (.tar.gz, with ovms binary and necessary libraries), in a ./dist directory.

*The release package is compatible with linux machines on which glibc version is greater than or equal to the build image version. For debugging, an image with a suffix -build is also generated (i.e. openvino/model_server-build:latest).*

Note: Images include OpenVINO 2021.1 release.



## 3. Starting Docker Container with a Single Model<a name="singlemodel"></a>

- When the models are ready and stored in correct folders structure, you are ready to start the Docker container with the 
OpenVINO&trade; model server. To enable just a single model, you _do not_ need any extra configuration file, so this process can be 
completed with just one command like below:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 openvino/model_server:latest \
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
      --file_system_poll_wait_seconds SECONDS
                                Time interval between config and model versions 
                                changes detection. Default is 1. Zero or negative
                                value disables changes monitoring.

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
                                their values, eg "{\"CPU_THROUGHPUT_STREAMS\": \"1\"}".
                                Default throughput streams for CPU and GPU are calculated by OpenVINO

```


- The model path could be local on docker container like mounted during startup or it could be Google Cloud Storage path 
in a format `gs://<bucket>/<model_path>`. In this case it will be required to 
pass GCS credentials to the docker container,
unless GKE kubernetes cluster, which handled the authorization automatically,
 is used.

- Below is an example presenting how to start docker container with a support for GCS paths to the models. The variable 
`GOOGLE_APPLICATION_CREDENTIALS` contain a path to GCP authentication key. 

```bash
docker run --rm -d  -p 9001:9001 ie-serving-py:latest \
-e GOOGLE_APPLICATION_CREDENTIALS=“${GOOGLE_APPLICATION_CREDENTIALS}”  \
-v ${GOOGLE_APPLICATION_CREDENTIALS}:${GOOGLE_APPLICATION_CREDENTIALS}
/ie-serving-py/start_server.sh ie_serving model --model_path gs://bucket/model_path --model_name my_model --port 9001
```

Learn [more about GCP authentication](https://cloud.google.com/docs/authentication/production).


- It is also possible to provide paths to models located in S3 compatible storage in a format `s3://<bucket>/<model_path>`. 

- In this case it is necessary to provide credentials to bucket by setting environmental variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. You can also set `AWS_REGION` variable, although it's not always required. 

- Incase of custom storage server compatible with S3, set `S3_ENDPOINT` environmental variable in a HOST:PORT format. 

- In an example below you can see how to start docker container serving single model located in S3.

```bash
docker run --rm -d  -p 9001:9001 openvino/model_server:latest \
-e AWS_ACCESS_KEY_ID=“${AWS_ACCESS_KEY_ID}”  \
-e AWS_SECRET_ACCESS_KEY=“${AWS_SECRET_ACCESS_KEY}”  \
-e AWS_REGION=“${AWS_REGION}”  \
-e S3_ENDPOINT=“${S3_ENDPOINT}”  \
--model_path  s3://bucket/model_path --model_name my_model --port 9001 --batch_size auto --model_version_policy '{"all": {}}'
```


If you need to expose multiple models, you need to create a model server configuration file, which is explained in the following section.

## 4. Starting docker container with a configuration file<a name="configfile"></a>

- Model server configuration file defines multiple models, which can be exposed for clients requests.
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
             "target_device": "HDDL"
         }
      }
   ]
}
```
**Note** :  Follow the below model repository structure for multiple models:

```bash
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

here the numerical values depict the version number of the model.


- It has a mandatory array `model_config_list`, which includes a collection of `config` objects for each served model. 
Each config object includes values for the model `name` and the `base_path` attributes.

- When the config file is present, the docker container can be started in a similar manner as a single model. Keep in mind that models with cloud storage path require specific environmental variables set. Configuration file above contains both GCS and S3 paths so starting docker container supporting all those models can be done with:

```bash
docker run --rm -d  -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001  -v <config.json>:/opt/ml/config.json ovms:latest \
--config_path /opt/ml/config.json --port 9001 --rest_port 8001
```


## 5. Starting docker container with Neural Compute Stick<a name="ncs"></a>

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

3. To start server with NCS you can use command similar to:

```
docker run --rm -it --net=host -u root --privileged -v /opt/model:/opt/model -v /dev:/dev -p 9001:9001 \
ovms:latest --model_path /opt/model --model_name my_model --port 9001 --target_device MYRIAD
```

`--net=host` and `--privileged` parameters are required for USB connection to work properly. 

`-v /dev:/dev` mounts USB drives.

A single stick can handle one model at a time. If there are multiple sticks plugged in, OpenVINO Toolkit 
chooses to which one the model is loaded. 

## 6. Starting docker container with HDDL<a name="hddl"></a>

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
ovms:latest --model_path /opt/model --model_name my_model --port 9001 --target_device HDDL --nireq 16
```

`--device=/dev/ion:/dev/ion` mounts the accelerator.

`-v /var/tmp:/var/tmp` enables communication with _hddldaemon_ running on the
 host machine

## 7. Starting docker container with GPU<a name="gpu"></a>

The GPU plugin uses the Intel® Compute Library for Deep Neural Networks (clDNN) to infer deep neural networks.
It employs for inference execution Intel® Processor Graphics including Intel® HD Graphics and Intel® Iris® Graphics

Before using GPU as OVMS target device, you need to install the required drivers. Next, start the docker container
with additional parameter `--device /dev/dri` to pass the device context and set OVMS parameter `--target_device GPU`.
The command example is listed below:

```
docker run --rm -it --device=/dev/dri -v /opt/model:/opt/model -p 9001:9001 \
ie-serving-py:latest /ie-serving-py/start_server.sh ie_serving model --model_path /opt/model --model_name my_model --port 9001 --target_device GPU
```

## 8. Starting Docker Container using Multi-Device Plugin<a name="multiplugin"></a>

If you have multiple inference devices available (e.g. Myriad VPUs and CPU) you can increase inference throughput by enabling the Multi-Device Plugin. 
With Multi-Device Plugin enabled, inference requests will be load balanced between multiple devices. 
For more detailed information read [OpenVino's Multi-Device plugin documentation](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_supported_plugins_MULTI.html).

In order to use this feature in OpenVINO&trade; Model Server, following steps are required:

Set target_device for the model in configuration json file to MULTI:<DEVICE_1>,<DEVICE_2> (e.g. MULTI:MYRIAD,CPU, order of the devices defines their priority, so MYRIAD devices will be used first in this example)

Below is exemplary config.json setting up Multi-Device Plugin for resnet model, using Intel® Movidius™ Neural Compute Stick and CPU devices:
```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "MULTI:MYRIAD,CPU"}
   }]
}
```
Starting OpenVINO&trade; Model Server with config.json (placed in ./models/config.json path) defined as above, and with grpc_workers parameter set to match nireq field in config.json:
```
docker run -d  --net=host -u root --privileged --rm -v $(pwd)/models/:/opt/ml:ro -v /dev:/dev -p 9001:9001 \
ovms-py:latest --config_path /opt/ml/config.json --port 9001 
```
Or alternatively, when you are using just a single model, start OpenVINO&trade; Model Server using this command (config.json is not needed in this case):
```
docker run -d  --net=host -u root --privileged --name ie-serving --rm -v $(pwd)/models/:/opt/ml:ro -v \
 /dev:/dev -p 9001:9001 ovms:latest model --model_path /opt/ml/resnet --model_name resnet --port 9001 --target_device 'MULTI:MYRIAD,CPU'
 ```
After these steps, deployed model will perform inference on both Intel® Movidius™ Neural Compute Stick and CPU.
Total throughput will be roughly equal to sum of CPU and Intel® Movidius™ Neural Compute Stick throughput.

## 9. Starting Docker Container Using Heterogeneous Plugin<a name="heteroplugin"></a>

[HETERO plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_HETERO.html) makes it possible to distribute a single inference processing and model between several AI accelerators.
That way different parts of the DL network can split and executed on optimized devices.
OpenVINO automatically divides the network to optimize the execution.

Similarly to the MULTI plugin, Heterogenous plugin can be configured by using `--target_device` parameter using the pattern: `HETERO:<DEVICE_1>,<DEVICE_2>`.
The order of devices defines their priority. The first one is the primary device while the second is the fallback.<br>
Below is a config example using heterogeneous plugin with GPU as a primary device and CPU as a fallback.

```json
{"model_config_list": [
   {"config": {
      "name": "resnet",
      "base_path": "/opt/ml/resnet",
      "batch_size": "1",
      "target_device": "HETERO:GPU,CPU"}
   }]
}
```
