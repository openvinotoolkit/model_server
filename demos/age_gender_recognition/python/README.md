# Age and Gender Recognition via REST API {#ovms_demo_age_gender_guide}
This article describes how to use OpenVINO&trade; Model Server to execute inference requests sent over the REST API interface. The demo uses a pretrained model from the [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) repository.

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip 

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

## Download the pretrained model for age and gender recognition
Download both components of the model (xml and bin file) using curl in the `model` directory

```console
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml -o model/1/age-gender-recognition-retail-0013.bin -o model/1/age-gender-recognition-retail-0013.xml

```

## Server Deployment
:::{dropdown} **Deploying with Docker**
Start OVMS container with image pulled in previous step and mount `model` directory :
```bash
docker run --rm -d -u $(id -u):$(id -g) -v $(pwd)/model:/models/age_gender -p 9000:9000 -p 8000:8000 openvino/model_server:latest --model_path /models/age_gender --model_name age_gender --port 9000 --rest_port 8000
```
:::
:::{dropdown} **Deploying on Bare Metal**
Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.
```bat
cd demos\age_gender_recognition\python
ovms --model_path model --model_name age_gender --port 9000 --rest_port 8000
```
:::
#### Requesting the Service
Clone the repository
```console
git clone https://github.com/openvinotoolkit/model_server.git
```

Enter age_gender_recognition python demo directory:
```console
cd model_server/demos/age_gender_recognition/python
```

Download sample image using the command :
```console
curl https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/2022.1.0/models/intel/age-gender-recognition-retail-0013/assets/age-gender-recognition-retail-0001.jpg -o age-gender-recognition-retail-0001.jpg
```

Install python dependencies:
```console
pip3 install -r requirements.txt
```
Run [age_gender_recognition.py](https://github.com/openvinotoolkit/model_server/blob/releases/2025/0/demos/age_gender_recognition/python/age_gender_recognition.py) script to make an inference:
```console
python age_gender_recognition.py --image_input_path age-gender-recognition-retail-0001.jpg --rest_port 8000
```
Sample Output :
```bash
age-gender-recognition-retail-0001.jpg (1, 3, 62, 62) ; data range: 0 : 239
{'outputs': {'prob': [[[[0.9874807]], [[0.0125193456]]]], 'age_conv3': [[[[0.25190413]]]]}}
```
Output format :
| Output Name      | Shape | Description |
| :---        |    :----   | :----    |
| age_conv3   | [1, 1, 1, 1] | Estimated age divided by 100 |
|prob | [1, 2, 1, 1] | Softmax output across 2 type classes [female, male] |
