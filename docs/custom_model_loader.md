         1         2         3         4         5         6         7         8         9
123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

Introduction:
============

Using custom loader user can intercept normal loading of the model in the model server 
and handle the model loading. The interface allows to return xml and weights as buffers 
to model server so that model server can use appropriate inference engine APIs to load 
the model.
This customer loader can be used to preprocess the models, implement licensing, implement 
model protection by saving encrypted models etc.

Custom Loader Interface:
========================
The custom loader interface conatins below components:
1. A section in config file to define custom loader
2. In model section, specification for using custom loader to load the model
3. A base custom loader class in OVMS with default interface functions

Writing a custom loader:
========================
1. Derive custom loader class from base class "CustomLoaderInterface"
2. Define the functionality for interface APIs:
		- loaderInit
		- loadModel
		- getModelBlacklistStatus
		- unloadModel
		- loaderDeInit
    refer to "src/customloaderinterface.hpp" for more details on these interface functions
3. Also define APIs to create the custom loader interface class (createCustomLoader_t) and 
   destroy the class (destroyCustomLoader_t)
4. Create a shared library of customloader and specify the same in configuration file.


Example Customloader:
=====================

An example custom loader is implemented under "src/example/SampleCustomLoader".

Follow the below steps to use example custom model loader:

Step 1: Prepare Docker*

Download the model server code & build the docker (make docker_build).
Once the docker is ready, create a folder where all the artefacts can be downloaded. Ensure the models, client 
components, images are all downloaded to this folder. Also create the json required into this folder.

< mkdir test_custom_loader >


Step 2: Download a Model

< curl --create-dirs https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.xml https://download.01.org/opencv/2020/openvinotoolkit/2020.4/open_model_zoo/models_bin/3/face-detection-retail-0004/FP32/face-detection-retail-0004.bin -o model/fdsample/1/face-detection-retail-0004.xml -o model/fdsample/1/face-detection-retail-0004.bin >

Ensure the model folder has required permisions


Step 3: Download the required Client Components

< curl https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_utils.py -o client_utils.py https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/face_detection.py -o face_detection.py  https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/client_requirements.txt -o client_requirements.txt >

< pip3 install -r client_requirements.txt >


Step 4: Download Data for Inference

< curl --create-dirs https://raw.githubusercontent.com/openvinotoolkit/model_server/master/example_client/images/people/people1.jpeg -o images/people1.jpeg >


Step 5: Prepare the config json.
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
                "base_path":"/sampleloader/model/fdsample",
                "custom_loader_options": {"loader_name":  "sampleloader", "xml_file":  "face-detection-retail-0004.xml", "bin_file": "face-detection-retail-0004.bin"}
                }
        }
        ]
}

Step 6: Start the model server container

< docker run -d -v ${PWD}/model:/sampleloader/model -v ${PWD}/sampleloader.json:/sampleloader/sampleloader.json -p 9000:9000 openvino/model_server:latest --config_path /sampleloader/sampleloader.json --port 9000  --log_level DEBUG >


Step 7: Run inference & Review the results

< python3 face_detection.py --batch_size 1 --width 300 --height 300 --input_images_dir images --output_dir results --model_name sampleloader-model >

Check the docker logs to see that custom loader is used to load the model. Also, it is implemented that after 2 minutes, the model gets unloaded and gets loaded back after another 2 minutes.
