# Prepare a Model Repository {#ovms_docs_models_repository}

```{toctree}
---
maxdepth: 1
hidden:
---

ovms_docs_cloud_storage
```

The AI models served by OpenVINO&trade; Model Server must be in either of the four formats:
- [OpenVINO IR](https://docs.openvino.ai/2023.2/openvino_docs_MO_DG_IR_and_opsets.html#doxid-openvino-docs-m-o-d-g-i-r-and-opsets), where the graph is represented in .bin and .xml files 
- [ONNX](https://onnx.ai/), using the .onnx file
- [PaddlePaddle](https://www.paddlepaddle.org.cn/en), using .pdiparams and .pdmodel files
- [TensorFlow](https://www.tensorflow.org/), using SavedModel, MetaGraph or frozen Protobuf formats.
- [TensorFlow Lite](https://www.tensorflow.org/lite), using the .tflite file

To use models trained in other formats you need to convert them first. To do so, use 
OpenVINO’s [Model Optimizer](https://docs.openvino.ai/2023.2/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for IR, or different
[converters](https://onnx.ai/supported-tools.html) for ONNX.

The models need to be placed and mounted in a particular directory structure and according to the following rules:
When the models are hosted on the cloud storage, they should be frozen to be imported successfully.

```
tree models/
models/
├── model1
│   ├── 1
│   │   ├── ir_model.bin
│   │   └── ir_model.xml
│   └── 2
│       ├── ir_model.bin
│       └── ir_model.xml
├── model2
│   └── 1
│       ├── ir_model.bin
│       ├── ir_model.xml
│       └── mapping_config.json
├── model3
│    └── 1
│        └── model.onnx
├── model4
│      └── 1
│        ├── model.pdiparams
│        └── model.pdmodel
├── model5
│      └── 1
│        ├── model.pdiparams
│        └── model.pdmodel
└── model6
       └── 1
         ├── variables
         └── saved_model.pb

``` 

- Each model should be stored in a dedicated directory, e.g. model1 and model2. 
- Each model directory should include a sub-folder for each of its versions (1,2, etc). The versions and their folder names should be positive integer values.  
**Note:** In execution, the versions are enabled according to a pre-defined version policy. If the client does not specify 
the version number in parameters, by default, the latest version is served.
- Every version folder _must_ include model files, that is, .bin and .xml for IR, .onnx for ONNX, .pdiparams and .pdmodel for Paddlepaddle. The file name can be arbitrary.

<<<<<<< HEAD
<<<<<<< HEAD
For more information on how to use cloud hosted models, refer to the [article](./using_cloud_storage.md).
=======
For more information on how to use cloud hosted models, refer to the [article](https://docs.openvino.ai/2023.2/ovms_docs_cloud_storage.html).
>>>>>>> a9ba16f3 (Reorganizing Deployment on a Local System section)
=======
For more information on how to use cloud hosted models, refer to the [article](./using_cloud_storage.md).
>>>>>>> cae2a9ee (Update docs/models_repository.md)
