# OpenVINO&trade; Model Server {#ovms_what_is_openvino_model_server}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   ovms_docs_quick_start_guide
   ovms_docs_starting_server
   ovms_docs_features
   ovms_docs_server_app
   ovms_docs_dag
   ovms_docs_binary_input
   ovms_docs_performance_tuning
   ovms_docs_advanced
   ovms_docs_demos
   ovms_docs_troubleshooting

@endsphinxdirective

OpenVINO&trade; Model Server (OVMS) is a high-performance system for serving models. Implemented in C++ for scalability and optimized for deployment on Intel architectures. Model server uses the same architecture and API as [TensorFlow Serving](https://github.com/tensorflow/serving) and [KServe](https://github.com/kserve/kserve) while applying OpenVINO for inference execution. Inference service is provided via gRPC or REST API, making deploying new algorithms and AI experiments easy.

![OVMS diagram](ovms_draft_diagram.png)

## Model Serving

A model server hosts models and makes them accessible to software components over standart network protocols. Functionally it works similarly to a web server: user sends a request and receives a response. 

![OVMS picture](ovms.png)

Model Server Advantages: 

- Ability to process several client requests simultaneously so that the model uses less device memory. 
- Inference can be run remotely, create a lightweight client with only the necessary functions to perform API calls for edge or cloud deployments. Bringing high execution performance even on low-end client hosts.
- Application can be written in any language which supports REST or gRPC calls and is independent from model framework, hardware or infrastructure. 
- Model topology and weight is not exposed directly to the client application, making it easier to control access to its IP.
- Model server is suitable for application architecture based on microservice.
- It is easy to deploy in cloud, Kubernetes and OpenShift. 
- Inference can be easily scaled horizontally or vertically, making it easy to add more loads without any changes in the application.


## OpenVINO Model Server Key Features: 

The models used by OpenVINO Model Server need to be stored locally or hosted remotely by object storage services. For more details, refer to [Preparing Model Storage](./models_repository.md) documentation.  

OpenVINO&trade; Model Server works with [Bare Metal Hosts](host.md) as well as [Docker containers](docker_container.md). It is also suitable for landing in the [Kubernetes environment](../deploy/README.md).

- Support for AI accelerators, such as 
[Intel Movidius Myriad VPUs](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_supported_plugins_MYRIAD.html), 
[GPU](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_supported_plugins_GPU.html), and 
[HDDL](https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_supported_plugins_HDDL.html) 
- [model reshaping](shape_batch_size_and_layout.md) in runtime for high-throughput and low-latency
- [directed Acyclic Graph Scheduler](dag_scheduler.md) - connecting multiple models to deploy complex processing solutions and reducing data transfer overhead
- [custom nodes in DAG pipelines](custom_node_development.md) - allowing model inference and data transformations to be implemented with a custom node C/C++ dynamic library
- [binary format of the input data](binary_input.md) - data can be sent in JPEG or PNG formats to reduce traffic and offload the client applications
- [serving stateful models](stateful_models.md) - models that operate on sequences of data and maintain their state between inference requests
- online deployment of new [model versions](model_version_policy.md)
- [configuration updates in runtime](online_config_changes.md) 
- [performance tuning](performance_tuning.md)
- [model caching](model_cache.md) - cache the models on first load and re-use models from cache on subsequent loads
- [metrics](metrics.md) - metrics compatible with Prometheus standard

A demonstration on how to use OpenVINO Model Server can be found in the [Quickstart guide](ovms_quickstart.md). 

## Additional Resources

* [Simplified Deployments with OpenVINO™ Model Server and TensorFlow Serving](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Simplified-Deployments-with-OpenVINO-Model-Server-and-TensorFlow/post/1353218) - learn how to perform inference on JPEG images using the gRPC API in OpenVINO Model Server

* [Inference Scaling with OpenVINO™ Model Server in Kubernetes and OpenShift Clusters](https://www.intel.com/content/www/us/en/developer/articles/technical/deploy-openvino-in-openshift-and-kubernetes.html) - scale inferencing with OpenVINO in Kubernetes and OpenShift using OpenVINO Model Server

* [Benchmarking results](https://docs.openvino.ai/2022.1/openvino_docs_performance_benchmarks_ovms.html) - see high performance gains on several public neural networks on multiple Intel® CPUs, GPUs and VPUs 

* [Speed and Scale AI Inference Operations Across Multiple Architectures](https://techdecoded.intel.io/essentials/speed-and-scale-ai-inference-operations-across-multiple-architectures/?elq_cid=3646480_ts1607680426276&erpm_id=6470692_ts1607680426276) - watch OVMS demo recording

* [Release Notes](https://github.com/openvinotoolkit/model_server/releases) - find out what’s new in the latest OpenVINO Model Server Release

## Contact OpenVINO Model Server Team

If you have a question, a feature request, or a bug report, feel free to submit a [Github issue](https://github.com/openvinotoolkit/model_server).
