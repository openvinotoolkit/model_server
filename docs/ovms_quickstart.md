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
--model_path /models/face-detection --model_name face-detection --port 9000 --log_level DEBUG --shape auto
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

That concludes the prediction using the model in IR format. It can be repeated for ONNX model like presented below in step 9.

### Step 9: Prediction Use Case with an ONNX model

Similar steps can be executed also for the model in ONNX format. Just the model files should be swapped.
Below is a complete functional use case.

Download the model:
```
curl -L --create-dir https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o resnet/1/resnet50-caffe2-v1-9.onnx
```

Start the OVMS container:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001
```

Get an image to classify:
```
wget -q https://github.com/openvinotoolkit/model_server/raw/main/example_client/images/bee.jpeg
```

Run inference requests with a client like presented below:
```python
def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

def getJpeg(path, size):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # format of data is HWC
    # add image preprocessing if needed by the model
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    #convert to NHWC
    img = img.transpose(2,0,1)
    # normalize to adjust to model training dataset
    img = preprocess(img)
    img = img.reshape(1,3,size,size)
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img

img1 = getJpeg('bee.jpeg', 224)

channel = grpc.insecure_channel("localhost:9001")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = "resnet"
request.inputs["gpu_0/data_0"].CopyFrom(make_tensor_proto(img1, shape=(img1.shape)))
result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs

output = make_ndarray(result.outputs["gpu_0/softmax_1"])
max = np.argmax(output)
print("Class is with highest score: {}".format(max))
print("Detected class name: {}".format(classes.imagenet_classes[max]))

```
It shows the following output:
```bash
bee.jpeg (1, 3, 224, 224) ; data range: -2.117904 : 2.64
Class is with highest score: 309
Detected class name: bee
```