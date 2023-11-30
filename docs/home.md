# OpenVINO&trade; Model Server {#ovms_what_is_openvino_model_server}

```{toctree}
---
maxdepth: 1
hidden:
---

ovms_docs_quick_start_guide
ovms_docs_serving_model
ovms_docs_deploying_server
ovms_docs_server_app
ovms_docs_features
ovms_docs_performance_tuning
```

Model Server hosts models and makes them accessible to software components over standard network protocols: a client sends a request to the model server, which performs model inference and sends a response back to the client. Model Server offers many advantages for efficient model deployment: 
- Remote inference enables using lightweight clients with only the necessary functions to perform API calls to edge or cloud deployments.
- Applications are independent of the model framework, hardware device, and infrastructure.
- Client applications in any programming language that supports REST or gRPC calls can be used to run inference remotely on the model server.
- Clients require fewer updates since client libraries change very rarely.
- Model topology and weights are not exposed directly to client applications, making it easier to control access to the model.
- Ideal architecture for microservices-based applications and deployments in cloud environments – including Kubernetes and OpenShift clusters.
- Efficient resource utilization with horizontal and vertical inference scaling.

![OVMS diagram](ovms_diagram.png)

## Serving with OpenVINO Model Server

OpenVINO&trade; Model Server (OVMS) is a high-performance system for serving models. Implemented in C++ for scalability and optimized for deployment on Intel architectures, the model server uses the same architecture and API as [TensorFlow Serving](https://github.com/tensorflow/serving) and [KServe](https://github.com/kserve/kserve) while applying OpenVINO for inference execution. Inference service is provided via gRPC or REST API, making deploying new algorithms and AI experiments easy.

![OVMS picture](ovms_high_level.png)

The models used by the server need to be stored locally or hosted remotely by object storage services. For more details, refer to [Preparing Model Repository](./models_repository.md) documentation. Model server works inside [Docker containers](deploying_server.md), on [Bare Metal](deploying_server.md), and in [Kubernetes environment](deploying_server.md).
Start using OpenVINO Model Server with a fast-forward serving example from the [Quickstart guide](ovms_quickstart.md) or explore [Model Server features](features.md).

## Additional Resources

* [Simplified Deployments with OpenVINO™ Model Server and TensorFlow Serving](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Simplified-Deployments-with-OpenVINO-Model-Server-and-TensorFlow/post/1353218) 
* [Inference Scaling with OpenVINO™ Model Server in Kubernetes and OpenShift Clusters](https://www.intel.com/content/www/us/en/developer/articles/technical/deploy-openvino-in-openshift-and-kubernetes.html) 
* [Benchmarking results](https://docs.openvino.ai/2023.2/openvino_docs_performance_benchmarks.html)
* [Speed and Scale AI Inference Operations Across Multiple Architectures Demo Recording](https://techdecoded.intel.io/essentials/speed-and-scale-ai-inference-operations-across-multiple-architectures/?elq_cid=3646480_ts1607680426276&erpm_id=6470692_ts1607680426276) 
* [Release Notes](https://github.com/openvinotoolkit/model_server/releases) 
