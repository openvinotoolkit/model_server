# String output model demo {#ovms_string_output_model_demo}
## Overview

This demo demonstrates example deployment of a model with output precision `ov::element::string`. The output text is serialized into corresponding fields in gRPC proto/REST body. This allows the client to consume the text directly and avoid the process of label mapping or detokenization.

### Download and prepare MobileNet model using from TensorFlow
The script below is downloading a public MobileNet model trained on the ImageNet data. The original model accepts on input the image array in the range of 0-1 and returns probabilities for all the trained classes. We are adding to the model preprocessing function changing the input data range to 0-255 and also postprocessing function which is retrieving the most likely label name as a string. 
This is a very handy functionality because if allows us to export the model with the included pre/post processing functions as the model layers. The client just receives the string data with the label name for the classified image.

```bash
pip install -r requirements.txt
python3 download_model.py

tree model
model
└── 1
    ├── assets
    ├── fingerprint.pb
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

### Start the OVMS container:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd):/workspace -p 8000:8000 openvino/model_server:latest \
--model_path /workspace/model --model_name image_net --rest_port 8000
```

## Send request
Use example client to send requests containing images via KServ REST API:
```bash
python3 image_classification_with_string_output.py 
```
Request may be sent also using other APIs (KServ GRPC, TFS). In this sections you can find short code samples how to do this:
- [TensorFlow Serving API](./clients_tfs.md)
- [KServe API](./clients_kfs.md)


## Expected output
Start processing:
```bash
Model name: image_net
../common/static/images/airliner.jpeg cassified as airliner
Iteration 0; Processing time: 10.54 ms; speed 94.88 fps

../common/static/images/arctic-fox.jpeg cassified as Arctic fox
Iteration 0; Processing time: 4.02 ms; speed 248.57 fps

../common/static/images/bee.jpeg cassified as bee
Iteration 0; Processing time: 3.23 ms; speed 309.50 fps

../common/static/images/golden_retriever.jpeg cassified as clumber
Iteration 0; Processing time: 3.83 ms; speed 261.03 fps

../common/static/images/gorilla.jpeg cassified as gorilla
Iteration 0; Processing time: 3.24 ms; speed 308.55 fps

../common/static/images/magnetic_compass.jpeg cassified as magnetic compass
Iteration 0; Processing time: 3.30 ms; speed 303.49 fps

../common/static/images/peacock.jpeg cassified as peacock
Iteration 0; Processing time: 3.36 ms; speed 297.89 fps

../common/static/images/pelican.jpeg cassified as pelican
Iteration 0; Processing time: 3.36 ms; speed 297.35 fps

../common/static/images/snail.jpeg cassified as snail
Iteration 0; Processing time: 3.13 ms; speed 319.59 fps

../common/static/images/zebra.jpeg cassified as zebra
Iteration 0; Processing time: 4.28 ms; speed 233.86 fps
```
