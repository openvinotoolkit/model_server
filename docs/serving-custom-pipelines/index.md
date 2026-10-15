---
id: ovms_docs_features
sidebar_position: 1
---

# Model Server Features

:::info Phase 1 structure note (pending stakeholder review)
Old catch-all 'Features' page repurposed as this section's landing page since most of its children moved here. PLACEHOLDER NOTE ADDED: IA explicitly says do not retain Features as a navigation container — Phase 2 must trim this page to pipeline-configuration content only, since several of its current links now point to pages that moved to Deploy-and-operate, Using-the-models, and Reference.
:::


## Efficient LLM Serving, text embeddings and reranking
Serve LLMs enhanced with state of the art optimization techniques for best performance and resource utilization on generative workloads

[Learn more](../using-the-models/genai-endpoints-overview.md)

## Python Code Execution
Write Python code that will do your custom processing and serve it in the Model Server.
Take advantage of a rich environment of Python modules in domains like data processing and data science to create flexible solutions without the need to write C++ code.

[Learn more](../reference/python-support-reference.md)

## Serving MediaPipe Graphs
Create [MediaPipe](https://developers.google.com/mediapipe/framework/framework_concepts/overview) graphs and serve them. Configure multiple nodes and connect them to create powerful pipelines.

[Learn more](mediapipe-graphs.md)

## Processing Raw Data
Send data in JPEG or PNG formats to reduce traffic and offload data pre-processing to the server.

[Learn more](binary-input-index.md)

## Model Versioning Policies
The model repository structure enables adding or deleting numerical version directories and the server will automatically adjust which models are served.
Control which model versions are served by setting the model version policy to serve all models, a specific model or set of models or just the latest version of the model (default setting).

[Learn more](../deploy-and-operate/model-version-policy.md)

## Model Reshaping
Change the batch size, shape and layout of the model at runtime to achieve high throughput and low latency.

[Learn more](shape-batch-size-and-layout.md)

## Modify Model Configuration at Runtime
OpenVINO Model Server regularly checks for changes to the configuration file and applies them during runtime. This means that you can change model configurations
(for example, change the device where a model is served), add a new model or completely remove one that is no longer needed. These changes will be applied without any disruption to the service.

[Learn more](../deploy-and-operate/online-config-changes.md)

## Idle Servable Management (Preview)
Unload inactive model and graph groups, then load them with the next inference request. Reduce CPU and GPU memory needed to serve multiple large models.

[Learn more](../deploy-and-operate/idle-servable-management.md)

## Metrics
Use the metrics endpoint compatible with the Prometheus to access performance and utilization statistics.

[Learn more](../reference/metrics.md)

## Enabling Dynamic Inputs
Configure served models to accept data with variable batch sizes and input shapes.

[Learn more](dynamic-input-index.md)

## Model Server C API
Use in process inference via model server to leverage the model management and model pipelines functionality of OpenVINO Model Server within an application. This allows to reuse existing OVMS functionality to execute inference locally without network overhead.

[Learn more](../using-the-models/c-api.md)

## Advanced Features
Use CPU Extensions, model cache feature or a custom model loader.

[Learn more](custom-nodes-and-extensions-index.md)
