# Prediction Example with an ONNX Model {#ovms_demo_using_onnx_model}

Steps are similar to when you work with IR model format. Model Server accepts ONNX models as well with no differences in versioning. Locate ONNX model file in separate model version directory.
Below is a complete functional use case using Python 3.7 or higher.
For this example let's use a public [ONNX ResNet](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet) model - resnet50-caffe2-v1-9.onnx model.

This model requires additional [preprocessing function](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet#preprocessing). Preprocessing can be performed in the client by manipulating data before sending the request, but a more efficient way is to delegate preprocessing to the server by setting preprocessing parameters. Server side preprocessing may have an impact on the performance by reducing amount of data sent, as `uint8` instead of `fp32`. More information about preprocessing parameters is available in [parameters.md](../../../docs/parameters.md).

## Adding preprocessing to the server side

Clone the repository and enter using_onnx_model directory

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/using_onnx_model/python
```

Prepare environment
```bash
pip install -r requirements.txt
curl --fail -L --create-dirs https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx -o workspace/resnet50-onnx/1/resnet50-caffe2-v1-9.onnx
```

Start the OVMS container with additional preprocessing options:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd)/workspace:/workspace -p 9001:9001 openvino/model_server:latest \
--model_path /workspace/resnet50-onnx --model_name resnet --port 9001 --layout NHWC:NCHW --mean "[123.675,116.28,103.53]" --scale "[58.395,57.12,57.375]" --shape "(1,224,224,3)" --color_format BGR:RGB --precision uint8:fp32
```

Run the client:
```bash
pip3 install -r requirements.txt
python onnx_model_demo.py --service_url localhost:9001
```
Output:
```
Running inference with image: ../../common/static/images/bee.jpeg
Class with highest score: 309
Detected class name: bee
```

The client can be also run with flag `--load_image` which loads input image as uint8. In this case the image needs to be resized and batch dimension needs to be added.
```bash
python onnx_model_demo.py --service_url localhost:9001 --load_image
```
Output:
```
Running inference with image: ../../common/static/images/bee.jpeg
Class with highest score: 309
Detected class name: bee
```
