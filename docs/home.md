# OpenVINO&trade; Model Server {#ovms_what_is_openvino_model_server}

```{toctree}
---
maxdepth: 1
hidden:
---

ovms_docs_quick_start_guide
ovms_docs_llm_quickstart
ovms_docs_models_repository
ovms_docs_deploying_server
ovms_docs_server_app
ovms_docs_features
ovms_docs_performance_tuning
ovms_docs_demos
ovms_docs_troubleshooting
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

OpenVINO&trade; Model Server (OVMS) is a high-performance system for serving models. Implemented in C++ for scalability and optimized for deployment on Intel architectures. It uses the same API as [OpenAI](./docs/genai.md), [Cohere](./docs/model_server_rest_api_rerank.md), [KServe](./docs/model_server_grpc_api_kfs.md) and [TensorFlow Serving](./docs/model_server_rest_api_tfs.md) and while applying OpenVINO for inference execution. Inference service is provided via gRPC or REST API, making deploying new algorithms and AI experiments easy.

Check how to write the client applications using [generative endpoints](./docs/clients_genai.md).

![OVMS picture](ovms_high_level.png)

The models used by the server need to be stored locally or hosted remotely by object storage services. For more details, refer to [Preparing Model Repository](./models_repository.md) documentation. Model server works inside [Docker containers](deploying_server.md), on [Bare Metal](deploying_server.md), and in [Kubernetes environment](deploying_server.md).
Start using OpenVINO Model Server with a fast-forward serving example from the [QuickStart guide](ovms_quickstart.md) or [LLM QuickStart guide](./llm/quickstart.md).

### Key features:
- **[NEW]** Native Windows support. Check updated [deployment guide](./deploying_server.md)
- **[NEW]** [Embeddings endpoint compatible with OpenAI API](../demos/embeddings/README.md)
- **[NEW]** [Reranking compatible with Cohere API](../demos/rerank/README.md)
- **[NEW]** [Efficient Text Generation with OpenAI API](../demos/continuous_batching/README.md)
- [Python code execution](python_support/reference.md)
- [gRPC streaming](streaming_endpoints.md)
- [MediaPipe graphs serving](mediapipe.md)
- Model management - including [model versioning](model_version_policy.md) and [model updates in runtime](online_config_changes.md)
- [Dynamic model inputs](shape_batch_size_and_layout.md)
- [Directed Acyclic Graph Scheduler](dag_scheduler.md) along with [custom nodes in DAG pipelines](custom_node_development.md)
- [Metrics](metrics.md) - metrics compatible with Prometheus standard
- Support for multiple frameworks, such as TensorFlow, PaddlePaddle and ONNX
- Support for [AI accelerators](./accelerators.md)

## Additional Resources
* [RAG building blocks made easy and affordable with OpenVINO Model Server](https://medium.com/openvino-toolkit/rag-building-blocks-made-easy-and-affordable-with-openvino-model-server-e7b03da5012b)
* [Simplified Deployments with OpenVINO™ Model Server and TensorFlow Serving](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Simplified-Deployments-with-OpenVINO-Model-Server-and-TensorFlow/post/1353218)
* [Inference Scaling with OpenVINO™ Model Server in Kubernetes and OpenShift Clusters](https://www.intel.com/content/www/us/en/developer/articles/technical/deploy-openvino-in-openshift-and-kubernetes.html)
* [Benchmarking results](https://docs.openvino.ai/2025/about-openvino/performance-benchmarks.html)
* [Release Notes](https://github.com/openvinotoolkit/model_server/releases)
