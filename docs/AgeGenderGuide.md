# Age and Gender Recognition via OpenVINO&trade; Model Server REST API

## Introduction
This documentation presents how to use OpenVINO&trade; Model Server to execute inference requests over REST API interface. The demo is using a pretrained model from [open_model_zoo](https://github.com/opencv/open_model_zoo) repository.

## Steps

#### Download the pretrained model for age and gender recognition
1. Download both component of the model(xml and bin file) using curl in `model` directory

```Bash
curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.bin https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml -o model/age-gender-recognition-retail-0013.bin -o model/age-gender-recognition-retail-0013.xml

```

#### Pull and tag the docker image with OVMS component 
1. Pull the latest version of OpenVINO&trade; Model Server from Dockerhub 
```Bash
docker pull openvino/model_server:latest
```

#### Start OVMS docker container with downloaded model
Start OVMS container with image pulled in previous step and mount `model` directory
```Bash 
docker run --rm -d -v $(pwd)/model:/models/age_gender/1 -p 9000:9000 -p 9001:9001 openvino/model_server:latest --model_path models/age_gender --model_name age_gender --port 9000 --rest_port 9001
```

####  Download Sample Image
Download sample image using command :
```Bash
wget https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0001.jpg
```

#### Format the json request and send the inference request to OVMS REST API endpoint
1. Create a sample python script using the command : 
```Bash
touch sample.py
```
2. Format the downloaded image using following python code snippet. Output of the code snippet is a json including the downloaded image in BGR format and 0-255 normalization. Paste the following code snippet in `sample.py`.
```Python
import cv2
import numpy as np
import json
import requests


def getJpeg(path, size):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # retrived array has BGR format and 0-255 normalization
    img = cv2.resize(img, (size, size))
    img = img.transpose(2,0,1).reshape(1,3,size,size)
    print(path, img.shape, "; data range:",np.amin(img),":",np.amax(img))
    return img

my_image = getJpeg('age-gender-recognition-retail-0001.jpg',62)


data_obj = {'inputs':  my_image.tolist()}
data_json = json.dumps(data_obj)

result = requests.post("http://localhost:9001/v1/models/age_gender:predict", data=data_json)
result_dict = json.loads(result.text)
print(result_dict)
```

3. Run the above code snippet to send POST API request to predict results by providing formatted json as request body using the command :
```Bash
python3 sample.py
```
Sample Output :
```
{'outputs': {'age_conv3': [[[[0.2519038915634155]]]], 'prob': [[[[0.9874807000160217]], [[0.012519358657300472]]]]}}
```
Output format :
| Output Name      | Shape | Description |
| :---        |    :----   | :----    |
| age_conv3   | [1, 1, 1, 1] | Estimated age divided by 100 |
|prob | [1, 2, 1, 1] | Softmax output across 2 type classes [female, male] |