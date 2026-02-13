# Prediction Example with an ONNX Model {#ovms_demo_using_onnx_model}

This demo demonstrates the steps required to deploy an ONNX‑based vision model. The workflow is optimized for rapid integration and ease of use: no model‑conversion step is needed, as the model is provided directly in ONNX format.
To further simplify deployment, the server applies all necessary image‑preprocessing operations, removing the need for the client to implement preprocessing pipelines such as normalization or color‑space transformation. This approach reduces development effort, ensures consistency with the model’s training configuration, and accelerates end‑to‑end deployment.
The server accepts image data in multiple formats, offering flexibility depending on the client environment. Images can be sent as:

Raw arrays directly obtained from OpenCV or Pillow
Encoded images, including JPEG or PNG formats

This enables seamless integration with a wide range of applications and client libraries.
Below is a complete functional use case using Python 3.7 or higher.
For this example let's use a public [ONNX ResNet](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet) model - resnet50-caffe2-v1-9.onnx model.

This model was trained using an additional [preprocessing](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet#preprocessing). For inference, preprocessing can be executed on the client side by transforming the input data before sending the request. However, a more efficient approach is to delegate preprocessing to the server by configuring the appropriate preprocessing parameters.
Here will be adjusted `mean`, `scale`, `color` and `layout`. In addition to that, input precision conversion from fp32 to uint8 can improve performance and bandwidth efficiency. Raw images can be transmitted using more compact uint8 data, significantly reducing the payload size and lowering client‑side compute requirements.
More details about [parameters](../../../docs/parameters.md).

## Adding preprocessing to the server side

Clone the repository and enter using_onnx_model directory

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/using_onnx_model/python
```

Prepare environment
```bash
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
