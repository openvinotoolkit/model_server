# Face Detection Demo {#ovms_demo_face_detection}

## Prerequisites

**Model preparation**: Python 3.9 or higher with pip 

**Model Server deployment**: Installed Docker Engine or OVMS binary package according to the [baremetal deployment guide](../../../docs/deploying_server_baremetal.md)

## Overview

The script [face_detection.py](https://github.com/openvinotoolkit/model_server/blob/releases/2025/3/demos/face_detection/python/face_detection.py) runs face detection inference requests for all the images
saved in `input_images_dir` directory.

The script can adjust the input image size and change the batch size in the request. It demonstrates how to use
the functionality of dynamic shape in OpenVINO Model Server and how to process the output from the server.

The example relies on the model [face-detection-retail-0004](https://github.com/openvinotoolkit/open_model_zoo/blob/releases/2022/1/models/intel/face-detection-retail-0004/README.md).

Clone the repository and enter face_detection directory
```console
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/face_detection/python
```

Prepare environment:
```console
pip install -r ../../common/python/requirements.txt
```

```console
python face_detection.py --help
usage: face_detection.py [-h] [--input_images_dir INPUT_IMAGES_DIR]
                         [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE]
                         [--width WIDTH] [--height HEIGHT]
                         [--grpc_address GRPC_ADDRESS] [--grpc_port GRPC_PORT]
                         [--model_name MODEL_NAME] [--tls]
                         [--server_cert SERVER_CERT]
                         [--client_cert CLIENT_CERT] [--client_key CLIENT_KEY]

Demo for face detection requests via TFS gRPC API analyses input images and
saves images with bounding boxes drawn around detected faces. It relies on
face_detection model...

optional arguments:
  -h, --help            show this help message and exit
  --input_images_dir INPUT_IMAGES_DIR
                        Directory with input images
  --output_dir OUTPUT_DIR
                        Directory for storing images with detection results
  --batch_size BATCH_SIZE
                        How many images should be grouped in one batch
  --width WIDTH         How the input image width should be resized in pixels
  --height HEIGHT       How the input image width should be resized in pixels
  --grpc_address GRPC_ADDRESS
                        Specify url to grpc service. default:localhost
  --grpc_port GRPC_PORT
                        Specify port to grpc service. default: 9000
  --model_name MODEL_NAME
                        Specify the model name
  --tls                 use TLS communication with gRPC endpoint
  --server_cert SERVER_CERT
                        Path to server certificate
  --client_cert CLIENT_CERT
                        Path to client certificate
  --client_key CLIENT_KEY
                        Path to client key
```

## Usage example

Start the OVMS service locally:

```console
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/1/face-detection-retail-0004.bin
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml -o model/1/face-detection-retail-0004.xml
```

## Deploying OVMS

:::{dropdown} **Deploying with Docker**
```bash
chmod -R 755 model
docker run --rm -d -u $(id -u):$(id -g) -v `pwd`/model:/models -p 9000:9000 openvino/model_server:latest --model_path /models --model_name face-detection --port 9000  --shape auto
```
:::
:::{dropdown} **Deploying on Bare Metal**
Assuming you have unpacked model server package, make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

as mentioned in [deployment guide](../../../docs/deploying_server_baremetal.md), in every new shell that will start OpenVINO Model Server.
```bat
ovms --model_path model --model_name face-detection --port 9000  --shape auto
```
:::
Run the client:
```console
mkdir results

python face_detection.py --batch_size 1 --width 300 --height 300 --grpc_port 9000

['people3.jpeg', 'people1.jpeg', 'people4.jpeg', 'people2.jpeg']
Start processing 4 iterations with batch size 1

Request shape (1, 3, 300, 300)
Response shape (1, 1, 200, 7)
image in batch item 0 , output shape (3, 300, 300)
detection 0 [[[0.         1.         0.9999211  0.45312274 0.1578588  0.5795723
   0.4229777 ]]]
x_min 135
y_min 47
x_max 173
y_max 126
detection 1 [[[0.         1.         0.99940693 0.22251067 0.22310418 0.34818068
   0.50511265]]]
x_min 66
y_min 66
x_max 104
y_max 151
detection 2 [[[0.         1.         0.98198867 0.61662686 0.08173735 0.740975
   0.40580124]]]
x_min 184
y_min 24
x_max 222
y_max 121
saving result to results/1_0.jpg
Iteration 1; Processing time: 28.69 ms; speed 34.85 fps

Request shape (1, 3, 300, 300)
Response shape (1, 1, 200, 7)
image in batch item 0 , output shape (3, 300, 300)
detection 0 [[[0.         1.         0.9999999  0.22627862 0.35042182 0.27032945
   0.43312052]]]
x_min 67
y_min 105
x_max 81
y_max 129
detection 1 [[[0.         1.         0.9999999  0.7980574  0.35572374 0.8422255
   0.42749226]]]
x_min 239
y_min 106
x_max 252
y_max 128
detection 2 [[[0.         1.         0.9999927  0.4413453  0.29417545 0.48191014
   0.37180012]]]
x_min 132
y_min 88
x_max 144
y_max 111
detection 3 [[[0.         1.         0.99964225 0.55356365 0.30400735 0.59468836
   0.38264883]]]
x_min 166
y_min 91
x_max 178
y_max 114
detection 4 [[[0.         1.         0.9993523  0.32912934 0.38222942 0.36873418
   0.44978413]]]
x_min 98
y_min 114
x_max 110
y_max 134
detection 5 [[[0.         1.         0.9992501  0.33522347 0.6249954  0.38323137
   0.7104612 ]]]
x_min 100
y_min 187
x_max 114
y_max 213
detection 6 [[[0.        1.        0.9976745 0.6488881 0.5992611 0.6988456 0.6907843]]]
x_min 194
y_min 179
x_max 209
y_max 207
detection 7 [[[0.        1.        0.9962077 0.5180316 0.5640176 0.5703776 0.6516389]]]
x_min 155
y_min 169
x_max 171
y_max 195
detection 8 [[[0.        1.        0.722986  0.6746904 0.3287916 0.7198625 0.4061382]]]
x_min 202
y_min 98
x_max 215
y_max 121
detection 9 [[[0.         1.         0.566281   0.13994813 0.36546633 0.18363091
   0.44829145]]]
x_min 41
y_min 109
x_max 55
y_max 134
saving result to results/2_0.jpg
Iteration 2; Processing time: 28.07 ms; speed 35.62 fps

Request shape (1, 3, 300, 300)
Response shape (1, 1, 200, 7)
image in batch item 0 , output shape (3, 300, 300)
detection 0 [[[0.         1.         0.99953306 0.1962457  0.08444536 0.33928737
   0.43830967]]]
x_min 58
y_min 25
x_max 101
y_max 131
detection 1 [[[0.         1.         0.9994074  0.7784856  0.12126188 0.9029822
   0.38237268]]]
x_min 233
y_min 36
x_max 270
y_max 114
saving result to results/3_0.jpg
Iteration 3; Processing time: 31.37 ms; speed 31.88 fps

Request shape (1, 3, 300, 300)
Response shape (1, 1, 200, 7)
image in batch item 0 , output shape (3, 300, 300)
detection 0 [[[0.         1.         0.99999976 0.46376142 0.10504608 0.62492824
   0.4838    ]]]
x_min 139
y_min 31
x_max 187
y_max 145
detection 1 [[[0.         1.         0.9998708  0.12117466 0.53293884 0.2551119
   0.753091  ]]]
x_min 36
y_min 159
x_max 76
y_max 225
saving result to results/4_0.jpg
Iteration 4; Processing time: 26.14 ms; speed 38.26 fps

processing time for all iterations
average time: 28.25 ms; average speed: 35.40 fps
median time: 28.00 ms; median speed: 35.71 fps
max time: 31.00 ms; min speed: 32.26 fps
min time: 26.00 ms; max speed: 38.46 fps
time percentile 90: 30.10 ms; speed percentile 90: 33.22 fps
time percentile 50: 28.00 ms; speed percentile 50: 35.71 fps
time standard deviation: 1.79
time variance: 3.19
```

```console
python face_detection.py --batch_size 4 --width 600 --height 400 --input_images_dir ../../common/static/images/people --output_dir results --grpc_port 9000

['people3.jpeg', 'people1.jpeg', 'people4.jpeg', 'people2.jpeg']
Start processing 1 iterations with batch size 4

Request shape (4, 3, 400, 600)
Response shape (1, 1, 800, 7)
image in batch item 0 , output shape (3, 400, 600)
detection 0 [[[0.         1.         1.         0.60533124 0.06616803 0.7431837
   0.39988554]]]
x_min 363
y_min 26
x_max 445
y_max 159
detection 1 [[[0.         1.         1.         0.4518694  0.14568813 0.5847517
   0.41192806]]]
x_min 271
y_min 58
x_max 350
y_max 164
detection 2 [[[0.         1.         1.         0.21670896 0.21642502 0.34729022
   0.4911797 ]]]
x_min 130
y_min 86
x_max 208
y_max 196
saving result to results/1_0.jpg
image in batch item 1 , output shape (3, 400, 600)
detection 47 [[[1.         1.         1.         0.55241716 0.30246916 0.59122956
   0.3917096 ]]]
x_min 331
y_min 120
x_max 354
y_max 156
detection 48 [[[1.         1.         0.9999999  0.33650026 0.6238751  0.38452044
   0.71090096]]]
x_min 201
y_min 249
x_max 230
y_max 284
detection 49 [[[1.         1.         0.99999917 0.22734313 0.34604052 0.2695081
   0.44246814]]]
x_min 136
y_min 138
x_max 161
y_max 176
detection 50 [[[1.         1.         0.99999905 0.4421842  0.29369104 0.4823516
   0.3778398 ]]]
x_min 265
y_min 117
x_max 289
y_max 151
detection 51 [[[1.         1.         0.9999931  0.51710206 0.56031024 0.5672712
   0.6503216 ]]]
x_min 310
y_min 224
x_max 340
y_max 260
detection 52 [[[1.         1.         0.99996173 0.79854035 0.35696298 0.8411697
   0.42877442]]]
x_min 479
y_min 142
x_max 504
y_max 171
detection 53 [[[1.         1.         0.99993885 0.65063936 0.59173524 0.6996588
   0.6923708 ]]]
x_min 390
y_min 236
x_max 419
y_max 276
detection 54 [[[1.         1.         0.99990237 0.67253107 0.3239646  0.7202968
   0.41418064]]]
x_min 403
y_min 129
x_max 432
y_max 165
detection 55 [[[1.         1.         0.99935323 0.13880333 0.36219725 0.18785533
   0.45266667]]]
x_min 83
y_min 144
x_max 112
y_max 181
detection 56 [[[1.         1.         0.9158124  0.3301806  0.37731093 0.3695439
   0.44581354]]]
x_min 198
y_min 150
x_max 221
y_max 178
saving result to results/1_1.jpg
image in batch item 2 , output shape (3, 400, 600)
detection 196 [[[2.         1.         1.         0.1979972  0.09549272 0.33667618
   0.43583226]]]
x_min 118
y_min 38
x_max 202
y_max 174
detection 197 [[[2.         1.         0.9994671  0.78329325 0.11969317 0.90646887
   0.38130802]]]
x_min 469
y_min 47
x_max 543
y_max 152
saving result to results/1_2.jpg
image in batch item 3 , output shape (3, 400, 600)
detection 299 [[[3.         1.         1.         0.45842466 0.11277443 0.6309127
   0.48450887]]]
x_min 275
y_min 45
x_max 378
y_max 193
detection 300 [[[3.         1.         0.999944   0.119257   0.5222848  0.2530746
   0.75714105]]]
x_min 71
y_min 208
x_max 151
y_max 302
saving result to results/1_3.jpg
Iteration 1; Processing time: 325.79 ms; speed 12.28 fps

processing time for all iterations
average time: 325.00 ms; average speed: 12.31 fps
median time: 325.00 ms; median speed: 12.31 fps
max time: 325.00 ms; min speed: 12.31 fps
min time: 325.00 ms; max speed: 12.31 fps
time percentile 90: 325.00 ms; speed percentile 90: 12.31 fps
time percentile 50: 325.00 ms; speed percentile 50: 12.31 fps
time standard deviation: 0.00
time variance: 0.00
```

The script will visualize the inference results on the images saved in the directory `output_dir`. Saved images have the
following naming convention:

```
[iteration]_[image_in_batch].jpeg
```
