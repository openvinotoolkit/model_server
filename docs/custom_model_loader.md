## Custom Model Loader {#ovms_docs_custom_loader}

Before loading the models directly from files, some extra processing or checking may be required. Typical examples are loading encrypted files or checking for model license, and other. In such cases, this customloader interface allows users to write their own custom model loader based on the predefine interface and load the same as a dynamic library. 

This document gives details on adding custom loader configuration, custom loader interface, and other details. 

## Custom Loader Interface

### Model Server Config File
A new section is added to the config file syntax to define customloader:

       "custom_loader_config_list":[
        {
                "config":{
                "loader_name": "#custom loader name",
                "library_path": "#Shared library path",
                "loader_config_file": "#Seperate config file with custom loader speicific details in json format"
                }
        }
        ]

Using the above syntax, multiple customloaders can be defined in the model server config file.

To enable a particular model to load using custom loader, add extra parameter in the model config as shown below:

        "model_config_list":[
        {
                "config":{
                "name":"sampleloader-model",
                "base_path":"model/fdsample",
                "custom_loader_options": {"loader_name":  "#custom loader name", "#parameters for customloader including file name etc in json format"}
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

An example customloader which reads files and returns required buffers to be loaded is implemented and provided as reference in **[src/example/SampleCustomLoader](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/example/SampleCustomLoader)**

This customloader is built with the model server build and available in the docker *openvino/model_server-build:latest*. The shared library can be either copied from this docker or built using makefile. An example Makefile is provided as  a reference in the directory.

## Running Example Customloader:

An example custom loader is implemented under "src/example/SampleCustomLoader".

Follow the below steps to use example custom model loader:

Step 1: Prepare test directory.

Download the model server code & build the docker (make docker_build).
Once the docker is ready, create a folder where all the artifacts can be downloaded. Make sure that the models, client components, and images are all downloaded to this folder. Additionally, create a json file required for this folder.
```
mkdir test_custom_loader
cd test_custom_loader
```

Step 2: Prepare the example of the custom loader library

```
cd model_server/src/example/SampleCustomLoader
make docker_build
```
It will generate the library in the `lib/libsampleloader.so` path.

Copy `lib` folder to the previously created directory `test_custom_loader`.

Step 3:  Download a Model

```
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/fdsample/1/face-detection-retail-0004.xml -o model/fdsample/1/face-detection-retail-0004.bin

chmod -R 755 ./model
```

Step 4: Download the required Client Components

```
curl --fail https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/face_detection/python/face_detection.py -o face_detection.py https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/python/requirements.txt -o requirements.txt

pip3 install -r requirements.txt
```


Step 5: Download Data for Inference

```
curl --fail --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2022/1/demos/common/static/images/people/people1.jpeg -o images/people1.jpeg
```

Step 6: Prepare the config json.

Example configuration file: Copy the following contents into a file and name it sampleloader.json

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
	}
	
Step 7: Start the model server container

```
docker run -d -v ${PWD}:/sampleloader -p 9000:9000 openvino/model_server:latest --config_path /sampleloader/sampleloader.json --port 9000  --log_level DEBUG
```

Step 8: Run inference & Review the results

```
python3 face_detection.py --batch_size 1 --width 300 --height 300 --input_images_dir images --output_dir results --model_name sampleloader-model
```

#### Blacklisting the model

Even though a model is specified in the config file, you may need to disable the model under certain conditions, for example, expired model license. 
To demonstrate this capability, this sample loader allows the users to specify an optional parameter "enable_file" in "custom_loader_options" in the configuration file. 
The file needs to be present at the model version folder (base path with the version number).

If you want to disable the model, create a file with the specified name and add a single line **DISABLED** to the file. 
The customloader checks for this file periodically and, if present with required string, marks the model for unloading. 
To reload the model either remove the string from the file or delete the file.
