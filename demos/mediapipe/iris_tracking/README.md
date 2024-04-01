# MediaPipe Iris Demo {#ovms_docs_demo_mediapipe_iris}

This guide shows how to implement [MediaPipe](../../../docs/mediapipe.md) graph using OVMS.

Example usage of graph that accepts Mediapipe::ImageFrame as a input:

The demo is based on the [Mediapipe Iris demo](https://github.com/google/mediapipe/blob/master/docs/solutions/iris.md)

## Prepare the server deployment

Clone the repository and enter mediapipe object_detection directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/mediapipe/iris_tracking

./prepare_server.sh

```
### Pull the Latest Model Server Image
Pull the latest version of OpenVINO&trade; Model Server from Docker Hub:
```Bash
docker pull openvino/model_server:latest

```

## Run OpenVINO Model Server
```bash
docker run -d -v $PWD/mediapipe:/mediapipe -v $PWD/ovms:/models -p 9000:9000 openvino/model_server:latest --config_path /models/config_iris.json --port 9000
```

## Run client application for iris tracking
```bash
pip install -r requirements.txt
# download a sample image for analysis
wget https://raw.githubusercontent.com/openvinotoolkit/model_server/main/demos/common/static/images/people/people2.jpeg
echo "people2.jpeg" > input_images.txt
# launch the client
python mediapipe_iris_tracking.py --grpc_port 9000 --images_list input_images.txt
Running demo application.
Start processing:
        Graph name: irisTracking
(800, 1200, 3)
Iteration 0; Processing time: 44.73 ms; speed 22.36 fps
Results saved to :image_0.jpg
```
## Output image
![output](output_image.jpg)
