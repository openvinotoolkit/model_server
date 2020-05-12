# OpenVINO&trade; Model Server Quickstart

To quickly start using OpenVINO Model Server please execute the following steps.

### Deploy Model Server

A Docker image with OpenVINO Model Server is available on DockerHub. Use the following command to download the latest release:

```bash
docker pull openvino/ubuntu18_model_server
```
In addition to the pre-built container image, see instructions to:
* [Build a container image with a Dockerfile](docker_container.md)
* [Build the application locally from source](host.md)

### Download a Model

OpenVINO Model Server requires a trained model to be able to perform an inference. The model
must be in IR format - a pair of files with .bin and .xml extensions. A model 
can be downloaded from various sites in the Internet (for example from the [Open Model Zoo](https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/) ) or converted from other formats - like TensorFlow, ONNX, Caffe or MXNet using [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) which is a part of the OpenVINO Toolkit.

Let's use the face detection model available under the following [link](https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/). 

Then download both components of the model to the `model` directory (for example using curl):
```
curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/face-detection-retail-0004.xml -o model/face-detection-retail-0004.bin
```

### Start the Model Server Container

To start the Model Server container use the following command:

```bash
docker run -d -v <folder_with_downloaded_model>:/models/face-detection/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server \
/ie-serving-py/start_server.sh ie_serving model --model_path /models/face-detection --model_name face-detection --port 9000  --shape auto
```

Model Server expects models in the predefined structure of folders - that is why the folder with downloaded models is mounted as `/models/face-detection/1`. In our case this expected structure is as follows:

```bash
models/
└── face-detection
    └── 1
        ├── face-detection-0106.bin
        └── face-detection-0106.xml

``` 

More about this structure and about how to deploy more than one model at the time can be found [here](./docker_container.md#preparing-the-models) and [here](./docker_container.md#starting-docker-container-with-a-configuration-file).

### Download the Example Client Script

The easiest way to access Model Server is using prepared client scripts. In our case we need to get a script that performs a face detection - like [this](../example_client/face_detection.py).

Together with this script we need to download a requirements.txt file with a list of libraries required to execute the script - this file is located [here](../example_client/client_requirements.txt) and the [file](https://github.com/openvinotoolkit/model_server/blob/master/example_client/client_utils.py) with some additional functions used by the client.

To download all this components we will use curl:

```
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/face_detection.py -o face_detection.py  https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_requirements.txt -o client_requirements.txt
```

More details about this particular script can be found [here](../example_client/face_detection.md). There is more scripts client scripts provided for Model Server - the
full list is [here](../example_client). 


### Download Data for Inference

Example images that can be used for inference can be downloaded from this [url](../example_client/images/people). Let's download the following [one](../example_client/images/people/people1.jpeg). The image should be located in the separate folder - as the script we are going to use makes an inference on all images in
a folder passed to it as a parameter.

```
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg
```

### Run Inference

After downloading the files mentioned earlier, go to the folder with test client script and execute the following commands:

* Install dependencies:
```
pip install -r client_requirements.txt
```

* Create the folder for results of the inference:

```
mdkir results
```

* Execute the script:

```
python face_detection.py --batch_size 1 --width 600 --height 400 --input_images_dir images --output_dir results
```

* Check the results

In the `results` folder there is an image with results of inference - the image given as input with faces in boxes.

