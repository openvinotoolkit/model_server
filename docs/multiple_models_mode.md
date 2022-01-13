# Deploy Multiple Models with a Config File {#ovms_docs_multiple_models}

### Starting Docker Container with a Configuration File for **Multiple** Models <a name="configfile"></a>

To use a container that has several models, you must use a model server configuration file that defines each model. The configuration file is in JSON format.
In the configuration file, provide an array, `model_config_list`, that includes a collection of config objects for each served model. For each config object include, at a minimum, values for the model name and the base_path attributes.

Example configuration file:
```json
{
   "model_config_list":[
      {
         "config":{
            "name":"model_name1",
            "base_path":"/opt/ml/models/model1",
            "batch_size": "16"
         }
      },
      {
         "config":{
            "name":"model_name2",
            "base_path":"/opt/ml/models/model2",
            "batch_size": "auto",
            "model_version_policy": {"all": {}}
         }
      },
      {
         "config":{
            "name":"model_name3",
            "base_path":"gs://bucket/models/model3",
            "model_version_policy": {"specific": { "versions":[1, 3] }},
            "shape": "auto"
         }
      },
      {
         "config":{
             "name":"model_name4",
             "base_path":"s3://bucket/models/model4",
             "shape": {
                "input1": "(1,3,200,200)",
                "input2": "(1,3,50,50)"
             },
         "plugin_config": {"CPU_THROUGHPUT_STREAMS": "CPU_THROUGHPUT_AUTO"}
         }
      },
      {
         "config":{
             "name":"model_name5",
             "base_path":"s3://bucket/models/model5",
             "shape": "auto",
             "nireq": 32,
             "target_device": "HDDL",
         }
      }
   ]
}
```

When the config file is present, the Docker container can be started similar to a single model. Keep in mind that models with cloud storage paths require setting specific environmental variables. Refer to the cloud storage requirements below for more details.

```bash

docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 -v <config.json>:/opt/ml/config.json openvino/model_server:latest \
--config_path /opt/ml/config.json --port 9001 --rest_port 8001

```

>NOTE: Follow the model repository structure below to serve multiple models:

```bash
models/
    model1
        1
         ir_model.bin
         ir_model.xml
        2
         ir_model.bin
         ir_model.xml
    model2
        1
         ir_model.bin
         ir_model.xml
         mapping_config.json
```

The numerical values represent the version number of the model.
