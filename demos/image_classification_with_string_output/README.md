# String output model demo {#ovms_string_output_model_demo}
## Overview

This demo demonstrates example deployment of a model with output precision `ov::element::string`. The output text is serialized into corresponding fields in gRPC proto/REST body. This allows the client to consume the text directly and avoid the process of label mapping or detokenization.

### Download and prepare MobileNet model using from TensorFlow
The script below is downloading a public MobileNet model trained on the ImageNet data. The original model accepts on input the image array in the range of 0-1 and returns probabilities for all the trained classes. We are adding to the model preprocessing function changing the input data range to 0-255 and also postprocessing function which is retrieving the most likely label name as a string. 
This is a very handy functionality because it allows us to export the model with the included pre/post processing functions as the model layers. The client just receives the string data with the label name for the classified image.

```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/image_classification_with_string_output
pip install -r requirements.txt
python download_model.py
rm model/1/fingerprint.pb

tree model
model
└── 1
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

### Start the OVMS container:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd):/workspace -p 8000:8000 openvino/model_server:latest \
--model_path /workspace/model --model_name mobile_net --rest_port 8000
```
Alternatively see (instructions)[https://github.com/openvinotoolkit/model_server/blob/main/docs/deploying_server_baremetal.md] for deployment on bare metal.

Make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

on every shell that will start OpenVINO Model Server.

And start Model Server using the following command:
```bat
ovms --model_name mobile_net --model_path model/ --rest_port 8000
```

## Send request
Use example client to send requests containing images via KServ REST API:
```console
python image_classification_with_string_output.py --http_port 8000
```
Request may be sent also using other APIs (KServ GRPC, TFS). In this sections you can find short code samples how to do this:
- [TensorFlow Serving API](../../docs/clients_tfs.md)
- [KServe API](../../docs/clients_kfs.md)


## Expected output
```console
Start processing:
        Model name: mobile_net
../common/static/images/airliner.jpeg classified  as airliner
Iteration 0; Processing time: 31.09 ms; speed 32.16 fps

../common/static/images/arctic-fox.jpeg classified  as Arctic fox
Iteration 0; Processing time: 5.27 ms; speed 189.75 fps

../common/static/images/bee.jpeg classified  as bee
Iteration 0; Processing time: 3.02 ms; speed 331.46 fps

../common/static/images/golden_retriever.jpeg classified  as clumber
Iteration 0; Processing time: 3.12 ms; speed 320.82 fps

../common/static/images/gorilla.jpeg classified  as gorilla
Iteration 0; Processing time: 3.04 ms; speed 329.06 fps

../common/static/images/magnetic_compass.jpeg classified  as magnetic compass
Iteration 0; Processing time: 3.10 ms; speed 323.00 fps

../common/static/images/peacock.jpeg classified  as peacock
Iteration 0; Processing time: 3.24 ms; speed 308.17 fps

../common/static/images/pelican.jpeg classified  as pelican
Iteration 0; Processing time: 3.17 ms; speed 315.36 fps

../common/static/images/snail.jpeg classified  as snail
Iteration 0; Processing time: 3.06 ms; speed 327.33 fps

../common/static/images/zebra.jpeg classified  as zebra
Iteration 0; Processing time: 4.19 ms; speed 238.72 fps
```
