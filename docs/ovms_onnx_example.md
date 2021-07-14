## Prediction Use Case Example with an ONNX Model

Similar steps like in [quick start](ovms_quickstart.md) with IR model format can be executed also for ONNX model. Just the model files should be swapped.
Below is a complete functional use case using python 3.6 or higher.

Download the model:
```bash
curl -L --create-dir https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o resnet/1/resnet50-caffe2-v1-9.onnx
```
```bash
tree resnet/
resnet/
└── 1
    └── resnet50-caffe2-v1-9.onnx
```
Note that the downloaded model requires additional [preprocessing function](https://github.com/onnx/models/tree/master/vision/classification/resnet#preprocessing).  

Start the OVMS container:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/resnet:/model -p 9001:9001 openvino/model_server:latest \
--model_path /model --model_name resnet --port 9001
```

Get an image to classify:
```
wget -q https://github.com/openvinotoolkit/model_server/raw/main/example_client/images/bee.jpeg
```
Install python libraries:
```bash
pip install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/client_requirements.txt
```
Get the list of imagenet classes:
```bash
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/example_client/classes.py
```

Run inference requests with a client like presented below:
```python
import numpy as nu

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