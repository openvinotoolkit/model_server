## Custom Model Loader {#ovms_docs_custom_loader}

### IMPORTANT: THIS FEATURE IS DEPRECATED AND WILL BE REMOVED IN THE FUTURE RELEASES 

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
A base class **CustomLoaderInterface** along with interface API is defined in [src/customloaderinterface.hpp](https://github.com/openvinotoolkit/model_server/blob/releases/2025/2/src/customloaderinterface.hpp)

Refer to this file  for API details. 

## Writing a Custom Loader:
Derive the new custom loader class from base class **CustomLoaderInterface** and define all the virtual functions specified. The library shall contain a function with name 

**CustomLoaderInterface* createCustomLoader**
which allocates the new custom loader and returns a pointer to the base class.

#### Blacklisting the model

Even though a model is specified in the config file, you may need to disable the model under certain conditions, for example, expired model license. 
To demonstrate this capability, this sample loader allows the users to specify an optional parameter "enable_file" in "custom_loader_options" in the configuration file. 
The file needs to be present at the model version folder (base path with the version number).

If you want to disable the model, create a file with the specified name and add a single line **DISABLED** to the file. 
The custom loader checks for this file periodically and, if present with required string, marks the model for unloading. 
To reload the model either remove the string from the file or delete the file.
