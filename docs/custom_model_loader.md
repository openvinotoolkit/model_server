## Custom Model Loader {#ovms_docs_custom_loader}

Before loading the models directly from files, some extra processing or checking may be required. Typical examples are loading encrypted files or checking for model license, and other. In such cases, this custom loader interface allows users to write their own custom model loader based on the predefined interface and load the same as a dynamic library. 

This document gives details on adding custom loader configuration, custom loader interface, and other details. 

## Custom Loader Interface

### Model Server Config File
A new section is added to the config file syntax to define custom loader:

       "custom_loader_config_list":[
        {
                "config":{
                "loader_name": "#custom loader name",
                "library_path": "#Shared library path",
                "loader_config_file": "#Separate config file with custom loader specific details in json format"
                }
        }
        ]

Using the above syntax, multiple custom loaders can be defined in the model server config file.

To enable a particular model to load using custom loader, add extra parameter in the model config as shown below:

        "model_config_list":[
        {
                "config":{
                "name":"sampleloader-model",
                "base_path":"model/fdsample",
                "custom_loader_options": {"loader_name":  "#custom loader name", "#parameters for custom loader including file name etc in json format"}
                }
        }
        ]


### C++ API Interface for custom loader:
A base class **CustomLoaderInterface** along with interface API is defined in [src/customloaderinterface.hpp](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/customloaderinterface.hpp)

Refer to this file  for API details. 

## Writing a Custom Loader:
Derive the new custom loader class from base class **CustomLoaderInterface** and define all the virtual functions specified. The library shall contain a function with name 

**CustomLoaderInterface* createCustomLoader**
which allocates the new custom loader and returns a pointer to the base class.

An example custom loader which reads files and returns required buffers to be loaded is implemented and provided as reference in **[src/example/SampleCustomLoader](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/example/SampleCustomLoader)**

This custom loader is built with the model server build and available in the docker *openvino/model_server-build:latest*. The shared library can be either copied from this docker or built using makefile. An example Makefile is provided as  a reference in the directory.

## Running Example Custom Loader:

An example custom loader is implemented under "src/example/SampleCustomLoader".

Follow the below steps to use example custom model loader:

Step 1: Prepare test directory.

Download the model server code & build the docker (make docker_build).
Once the docker is ready, create a folder where all the artifacts can be downloaded. Make sure that the models, client components, and images are all downloaded to this folder. Additionally, create a json file required for this folder.
```bash
mkdir test_custom_loader
```

Step 2: Prepare the example of the custom loader library

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/src/example/SampleCustomLoader
make docker_build
```
It will generate the library in the `lib/libsampleloader.so` path.

Copy `lib` folder to the previously created directory `test_custom_loader`.
```bash
cp -r lib ../../../../test_custom_loader/lib
cd ../../../../test_custom_loader
```

Step 3:  Download a Model

```bash
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/fdsample/1/face-detection-retail-0004.xml -o model/fdsample/1/face-detection-retail-0004.bin

chmod -R 755 ./model
```

Step 4: Download the required Client Components

```bash
curl --fail https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/3/demos/common/python/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/3/demos/face_detection/python/face_detection.py -o face_detection.py https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/3/demos/common/python/requirements.txt -o requirements.txt

pip3 install --upgrade pip
pip3 install -r requirements.txt
```


Step 5: Download Data for Inference

```bash
curl --fail --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/people/people1.jpeg -o images/people1.jpeg
```

Step 6: Prepare the config json.

Example configuration file: Create a sampleloader.json file:
```bash
echo '	
		{
	        "custom_loader_config_list":[
	        {
	                "config":{
	                "loader_name":"sampleloader",
	                "library_path": "/sampleloader/lib/libsampleloader.so",
	                "loader_config_file": "config.json"
	                }
	        }
	        ],
	        "model_config_list":[
	        {
	                "config":{
	                "name":"sampleloader-model",
	                "base_path":"/sampleloader/model/fdsample",
	                "custom_loader_options": {"loader_name":  "sampleloader", "model_file":  "face-detection-retail-0004.xml", "bin_file": "face-detection-retail-0004.bin", "enable_file": "face-detection-retail.status"}
	                }
	        }
	        ]
	}' >> sampleloader.json
```

Step 7: Start the model server container

```bash
docker run -d -v ${PWD}:/sampleloader -p 9000:9000 openvino/model_server:latest --config_path /sampleloader/sampleloader.json --port 9000  --log_level DEBUG
```

Step 8: Run inference & Review the results

```bash
python3 face_detection.py --batch_size 1 --width 300 --height 300 --input_images_dir images --output_dir results --model_name sampleloader-model --grpc_port 9000

['people1.jpeg']
Start processing 1 iterations with batch size 1

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
saving result to results/1_0.jpg
Iteration 1; Processing time: 21.92 ms; speed 45.61 fps

processing time for all iterations
average time: 21.00 ms; average speed: 47.62 fps
median time: 21.00 ms; median speed: 47.62 fps
max time: 21.00 ms; min speed: 47.62 fps
min time: 21.00 ms; max speed: 47.62 fps
time percentile 90: 21.00 ms; speed percentile 90: 47.62 fps
time percentile 50: 21.00 ms; speed percentile 50: 47.62 fps
time standard deviation: 0.00
time variance: 0.00
```

#### Blacklisting the model

Even though a model is specified in the config file, you may need to disable the model under certain conditions, for example, expired model license. 
To demonstrate this capability, this sample loader allows the users to specify an optional parameter "enable_file" in "custom_loader_options" in the configuration file. 
The file needs to be present at the model version folder (base path with the version number).

If you want to disable the model, create a file with the specified name and add a single line **DISABLED** to the file. 
The custom loader checks for this file periodically and, if present with required string, marks the model for unloading. 
To reload the model either remove the string from the file or delete the file.
