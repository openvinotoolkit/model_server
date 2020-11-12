##Introduction:

At times, before loading the models directly from files, some extra processing or checking may be required. Typical examples are loading encrypted files or checking for license for the model etc. In those cases,  this customloader interface allows users to write their own custom model loader based on the predefine interface and load the same as a dynamic library. 

This document gives details on adding custom loader configuration, custom loader interface and other details. 

##Custom Loader Interface:
###Model Server Config File
A new section is added to config file syntax to define customloader. The section will be:

       "custom_loader_config_list":[
        {
                "config":{
                "loader_name": "#custom loader name",
                "library_path": "#Shared library path",
                "loader_config_file": "#Seperate config file with custom loader speicific details in json format"
                }
        }

Using the above syntax multiple customloaders can be define in the model server config file.

To specifically enable a particular model to be loaded using customer loader, add extra parameter in model config as shown below:

        "model_config_list":[
        {
                "config":{
                "name":"sampleloader-model",
                "base_path":"model/fdsample",
                "custom_loader_options": {"loader_name":  "#custom loader name", "#parameters for customloader including file name etc in json format"}
                }
        }
        ]

###C++ API Interface for custom loader:
A base class **CustomLoaderInterface** along with interface API is defined in src/customloaderinterface.hpp 

Refer to the this file  for API details. 

##Writing a Custom Loader:
Derive the new custom loader class from base class **"CustomLoaderInterface"** and define all the virtual functions specified. The library shall contain a function with name 
**CustomLoaderInterface* createCustomLoader**
which allocates the new custom loader and return a pointer to the base class.

An example customloader which reads files and returns required buffers to be loaded is implemented and provided as reference in ** src/example/SampleCustomLoader **

This customloader is build with model server build. However an example Makefile is provided as reference also.

##Running Example Customloader:

An example custom loader is implemented under "src/example/SampleCustomLoader".

Follow the below steps to use example custom model loader:

Step-1: Prepare Docker

Download the model server code & build the docker (make docker_build).
Once the docker is ready, create a folder where all the artefacts can be downloaded. Ensure the models, client components, images are all downloaded to this folder. Also create the json required into this folder.
```
mkdir test_custom_loader
cd test_custom_loader
```

Step-2:  Download a Model

```
curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/fdsample/1/face-detection-retail-0004.xml -o model/fdsample/1/face-detection-retail-0004.bin
```

Step-3: Download the required Client Components

```
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/face_detection.py -o face_detection.py  https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_requirements.txt -o client_requirements.txt

pip3 install -r client_requirements.txt
```


Step-4: Download Data for Inference

```
curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg
```

Step-5: Prepare the config json.

Example configuration file: Copy the following contents into a file and name it sampleloader.json

	{
	        "custom_loader_config_list":[
	        {
	                "config":{
	                "loader_name":"sampleloader",
	                "library_path": "/ovms/custom_loader/libsampleloader.so",
	                "loader_config_file": "config.json"
	                }
	        }
	        ],
	        "model_config_list":[
	        {
	                "config":{
	                "name":"sampleloader-model",
	                "base_path":"model/fdsample",
	                "custom_loader_options": {"loader_name":  "sampleloader", "model_file":  "face-detection-retail-0004.xml", "bin_file": "face-detection-retail-0004.bin", "enable_file": "face-detection-retail.status"}
	                }
	        }
	        ]
	}
	
Step-6: Start the model server container

```
docker run -d -v ${PWD}/model:/sampleloader/model -v ${PWD}/sampleloader.json:/sampleloader/sampleloader.json -p 9000:9000 openvino/model_server:latest --config_path /sampleloader/sampleloader.json --port 9000  --log_level DEBUG 
```

Step-7: Run inference & Review the results

```
< python3 face_detection.py --batch_size 1 --width 300 --height 300 --input_images_dir images --output_dir results --model_name sampleloader-model >
```

####A note on blacklisting the model:
Even though a model is specified the config file, based on certain conditions, for example license expiry, the model may needs to be disabled. To demonstrate this capability, this sample loader allows the users to specify an optional parameter "enable_file" in "custom_loader_options" in configuration file. The file needs to be present at the same base path as models

Incase user want to black list the model, create the file specified with a single line **DISABLED**. The customloader checks for this file periodically and if present with required string marks the model for unloading. To load the model back either removed this string or remove the file.