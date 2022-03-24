# Quickstart Guide {#ovms_docs_quick_start_guide}

OpenVINO Model Server can perform inference using pre-trained models in either [OpenVINO IR](https://docs.openvino.ai/2022.1/openvino_docs_MO_DG_IR_and_opsets.html#doxid-openvino-docs-m-o-d-g-i-r-and-opsets) 
or [ONNX](https://onnx.ai/) format. You can get them by:

- downloading proper models from [Open Model Zoo](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/)
- converting other formats using [Model Optimizer](https://docs.openvino.ai/2022.1/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

To quickly start using OpenVINO™ Model Server follow these steps:
1. Prepare Docker
2. Download or build the OpenVINO™ Model server
3. Provide a model
4. Start the Model Server Container
5. Prepare the Example Client Components
6. Download data for inference
7. Run inference
8. Review the results


### Step 1: Prepare Docker

[Install Docker Engine](https://docs.docker.com/engine/install/), including its [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/), on your development system. 
If you are not sure if it has already been installed, test it using the following command. If it displays a test image and a message, it is ready.

``` bash
$ docker run hello-world
``` 

### Step 2: Download or Build the OpenVINO Model Server

Download the Docker image that contains OpenVINO Model Server available through DockerHub:

```bash
docker pull openvino/model_server:latest
```

or build the openvino/model_server:latest docker image, using:

```bash
make docker_build
```

### Step 3: Provide a model

Store components of the downloaded or converted model in the `model/1` directory. Here is an example command using curl and a face detection model:

```
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/1/face-detection-retail-0004.xml -o model/1/face-detection-retail-0004.bin
```

**Note:** For ONNX models additional steps are required. For a detailed description refer to our [ONNX format example](../demos/using_onnx_model/python/README.md).


### Step 4: Start the Model Server Container

Start the Model Server container:

```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/model:/models/face-detection -p 9000:9000 openvino/model_server:latest \
--model_path /models/face-detection --model_name face-detection --port 9000 --plugin_config '{"CPU_THROUGHPUT_STREAMS": "1"}' --shape auto
```

OpenVINO Model Server expects a particular folder structure  for models -  in this case it will be mounted as: 
`/models/face-detection/1`

```bash
models/
└── face-detection
    └── 1
        ├── face-detection-retail-0004.bin
        └── face-detection-retail-0004.xml
``` 

For more information on the folder structure and how to deploy more than one model at a time, check these links:
- [Prepare models](models_repository.md)
- [Deploy multiple models at once and to start a Docker container with a configuration file](multiple_models_mode.md)


### Step 5: Prepare the Example Client Components

Model scripts are available to provide an easy way to access Model Server. Here is an example command, using face detection and curl, to download all necessary components:

```
curl --fail https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/face_detection/python/face_detection.py -o face_detection.py https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/requirements.txt -o client_requirements.txt
```

For more information, check these links:

- [Information on the face detection script](../demos/face_detection/python/README.md). 
- [More Model Server client scripts](../demos/README.md).

### Step 6: Download Data for Inference

Provide inference data by putting the files in a separate folder, as inference will be performed on all files contained in it.

In this case, you can download [example images for inference](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/demos/common/static/images/people). This example uses a file named [people1.jpeg](https://github.com/openvinotoolkit/model_server/tree/releases/2022/1/demos/common/static/images/people/people1.jpeg) 
and use a single people1.jpeg file to run the following script:

```
curl --fail --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/people/people1.jpeg -o images/people1.jpeg
```

### Step 7: Run Inference

Go to the folder with the client script and install dependencies. Create a folder where inference results will be put and run the client script. For example:

```
pip install -r client_requirements.txt

mkdir results

python face_detection.py --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results
```

### Step 8: Review the Results

In the `results` folder, you can find files containing inference results. 
In our case, it will be a modified input image with bounding boxes indicating detected faces.

Note: Similar steps can be repeated also for the model with ONNX model. Check the inference [use case example with a public ResNet model in ONNX format](../demos/using_onnx_model/python/README.md). 
