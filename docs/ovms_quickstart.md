# OpenVINO&trade; Model Server Quickstart

The OpenVINO Model Server requires a trained model in Intermediate Representation (IR) or ONNX format on which it performs inference. Options to download appropriate models include:
 
- Downloading models from the [Open Model Zoo](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/)
- Using the [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert models to the IR format from formats like TensorFlow*, ONNX*, Caffe*, MXNet* or Kaldi*.

This guide uses the [face detection model](https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/). 

Use the steps in this guide to quickly start using OpenVINO™ Model Server. In these steps, you:

- Prepare Docker*
- Download and build the OpenVINO™ Model server
- Download a model
- Start the model server container
- Download the example client components
- Download data for inference
- Run inference
- Review the results

### Step 1: Prepare Docker

To see if you have Docker already installed and ready to use, test the installation:

``` bash
$ docker run hello-world
``` 

If you see a test image and an informational message, Docker is ready to use. Go to [download and build the OpenVINO Model Server](#step-2-download-and-build-the-openvino-model-server). 
If you don't see the test image and message:

1. [Install the Docker* Engine on your development machine](https://docs.docker.com/engine/install/).
2. [Use the Docker post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/).

Continue to Step 2 to download and build the OpenVINO Model Server.

### Step 2: Download and Build the OpenVINO Model Server

1. Download the Docker* image that contains the OpenVINO Model Server. This image is available from DockerHub:

```bash
docker pull openvino/model_server:latest
```
or build the docker image openvino/model_server:latest with a command:

```bash
make docker_build
```

### Step 3: Download a Model in IR Format

Download the model components to the `model/1` directory. Example command using curl:

```
curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/1/face-detection-retail-0004.xml -o model/1/face-detection-retail-0004.bin
```

### Step 4: Start the Model Server Container

Start the Model Server container:

```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/model:/models/face-detection -p 9000:9000 openvino/model_server:latest \
--model_path /models/face-detection --model_name face-detection --port 9000 --plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}' --shape auto
```

The Model Server expects models in a defined folder structure. The folder with the models is mounted as `/models/face-detection/1`, such as:

```bash
models/
└── face-detection
	└── 1
		├── face-detection-retail-0004.bin
		└── face-detection-retail-0004.xml
``` 


Use these links for more information about the folder structure and how to deploy more than one model at the time: 
- [Prepare models](./models_repository.md#preparing-the-models-repository)
- [Deploy multiple models at once and to start a Docker container with a configuration file](./docker_container.md#step-3-start-the-docker-container)

### Step 5: Download the Example Client Components

Model scripts are available to provide an easy way to access the Model Server. This example uses a face detection script and uses curl to download components.

1. Use this command to download all necessary components:

```
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/face_detection.py -o face_detection.py  https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_requirements.txt -o client_requirements.txt
```

For more information:

- [Information about the face detection script](../example_client/face_detection.md). 
- [More Model Server client scripts](../example_client).


### Step 6: Download Data for Inference

1. Download [example images for inference](../example_client/images/people). This example uses a file named [people1.jpeg](../example_client/images/people/people1.jpeg). 
2. Put the image in a folder by itself. The script runs inference on all images in the folder.

```
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg
```

### Step 7: Run Inference

1. Go to the folder in which you put the client script.

2. Install the dependencies:

```
pip install -r client_requirements.txt
```

3. Create a folder in which inference results will be put:

```
mkdir results
```

4. Run the client script:

```
python face_detection.py --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results
```

### Step 8: Review the Results

In the `results` folder, look for an image that contains the inference results. 
The result is the modified input image with bounding boxes indicating detected faces.


Note: Similar steps can be repeated also for the model with ONNX model. Check the inference [use case example with a public ResNet model in ONNX format](ovms_onnx_example.md). 