# Multiple-Model mode with a Config File {#ovms_docs_multiple_models}

To use a container that has several models, you need an additional JSON configuration file defining each of them. In the file, provide an array, 
`model_config_list`, that includes a collection of config objects for each served model. Every such config object needs the minimum of 
the name and the base_path attribute values provided.

An example of the configuration file:
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
             "target_device": "HDDL"
         }
      }
   ]
}
```

When the Docker container has the config file mounted, it can be started - the command is minimalistic, as arguments are read from the config file. 
Note that models with a cloud storage path require setting specific environmental variables.

```bash

docker run --rm -d -v /models/:/opt/ml:ro -p 9001:9001 -p 8001:8001 -v <config.json>:/opt/ml/config.json openvino/model_server:latest \
--config_path /opt/ml/config.json --port 9001 --rest_port 8001

```

For additional details, refer to:
- [Model Repository](models_repository.md)
- [Using Cloud Storage](using_cloud_storage.md)
