# Serving Pipelines of Models {#ovms_docs_dag}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_demuliplexing
   ovms_docs_dag_config
   ovms_docs_custom_node_development


@endsphinxdirective

## Introduction

The Directed Acyclic Graph (DAG) Scheduler makes it possible to create a pipeline of models for execution in a single client request. The pipeline is a Directed Acyclic Graph with different nodes that define how to process each step of a predict request. 

Using a pipeline, there is no need to return intermediate results of every model to the client. This allows avoiding network overhead by minimizing the number of requests sent to the Model Server. Each model output can be mapped to another model input. Since intermediate results are kept in the server's RAM, they can be reused by subsequent inferences, which reduces overall latency.

* <a href="#using-pipelines">Using the pipelines</a>
* <a href="#pipeline-examples">Pipelines examples </a>
* <a href="#node-type">Node Types</a>
* <a href="#current-limitations">Current Limitations</a>


## Using Pipelines <a name="using-pipelines"></a>

Pipelines can use the same API as the models. There are exactly the same calls for running 
the predictions. The request format must match the pipeline definition inputs.

The pipeline configuration can be queried using [gRPC GetModelMetadata](model_server_grpc_api_tfs.md) calls and
[REST Metadata](model_server_rest_api_tfs.md).
It returns the definition of the pipelines inputs and outputs. 

Similarly, pipelines can be queried for their state using the calls [GetModelStatus](model_server_grpc_api_tfs.md)
and [REST Model Status](model_server_rest_api_tfs.md).

The only difference in using the pipelines and individual models is in version management. In all calls to the pipelines, 
the version parameter is ignored. Pipelines are not versioned. Though, they can reference a particular version of the models in the graph.

## Pipelines Examples <a name="pipeline-examples"></a>

[Single face analysis with combined models](../demos/single_face_analysis_pipeline/python/README.md)

[Multiple vehicles analysis using demultiplexer with model_zoo_object_detection example custom node](../demos/vehicle_analysis_pipeline/python/README.md)

[Optical Character Recognition pipeline with east_ocr example custom node](../demos/optical_character_recognition/python/README.md)

[Horizontal Text Detection pipeline with horizontal_ocr example custom node](../demos/horizontal_text_detection/python/README.md)

## Node Types <a name="node-type"></a>

### Auxiliary Node Types
There are two special kinds of nodes - Request and Response node. Both of them are predefined and included in every pipeline definition you create:
*  Request node
    - This node defines which inputs are required to be sent via gRPC/REST request for pipeline usage. You can refer to it by node name: `request`.
* Response node
    - This node defines which outputs will be fetched from the final pipeline state and packed into gRPC/REST response. 
    You cannot refer to it in your pipeline configuration since it is the pipeline final stage. To define final outputs fill `outputs` field. 

### Deep Learning node type

* DL model - this node contains underlying OpenVINO&trade; model and performs inference on the selected target device. This can be defined in the configuration file. 
    Each model input needs to be mapped to some node's `data_item` - input from gRPC/REST `request` or another `DL model` output. 
    Outputs of the node may be mapped to another node's inputs or the `response` node, meaning it will be exposed in gRPC/REST response. 

### Custom node type

* custom - that node can be used to implement all operations on the data which can not be handled by the neural network model. It is represented by
a C++ dynamic library implementing OVMS API defined in [custom_node_interface.h](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/src/custom_node_interface.h). Custom nodes can run the data
processing using OpenCV, which is included in OVMS, or include other third-party components. Custom node libraries are loaded into OVMS
 by adding their definition to the pipeline configuration. The configuration includes a path to the compiled binary with the `.so` extension. 
Custom nodes are not versioned, meaning one custom node library is bound to one name. To load another version, another name needs to be used.

    OpenVINO Model Server docker image comes with prebuilt custom nodes that you can use out-of-the-box in your pipeline. See the list of built-in custom nodes and
    learn more about developing custom nodes yourself in the [custom node developer guide](custom_node_development.md).

## Demultiplexing data

During the pipeline execution, it is possible to split a request with multiple batches into a set of branches with a single batch.
That way a model configured with a batch size 1, can process requests with arbitrary batch size. Internally, OVMS demultiplexer will
divide the data, process them in parallel and combine the results. 

De-multiplication of the node output is enabled in the configuration file by adding `demultiply_count`. 
It assumes the batches are combined on the first dimension which is dropped after splitting. For example:
- a node returns output with shape `[8,1,3,224,224]`
- demuliplexer creates 8 requests with shape `[1,3,224,224]`
- next model processes in parallel 8 requests with output shape `[1,1001]` each.
- results are combined into a single output with shape `[8,1,1001]`

[Learn more about demuliplexing](demultiplexing.md) 


## Current limitations <a name="current-limitations"></a>

- Models with "auto" [batch size](dynamic_bs_auto_reload.md) or [shape](dynamic_shape_auto_reload.md) cannot be referenced in pipeline
- Connected inputs and output for subsequent node models need to match each other in terms of data shape, precision and layout - 
there is no automatic conversion between input/output model precisions or layouts. This limitation can be addressed with `--shape` and `--layout` model configuration or with a custom node to transform the data as required to match the expected data format.
- REST requests with no named format (JSON body with one unnamed input) are not supported


## See Also

- [Optimization of Performance](./performance_tuning.md)
