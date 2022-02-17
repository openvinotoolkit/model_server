# Helm Deployment {#ovms_deploy_helm_chart}

To simplify deployment in Kubernetes, we provide a helm chart for installing OpenVINO Model Server in a Kubernetes cluster. 
The helm chart is managing the Model Server instance which represents a kubernetes deployment and a
kubernetes service with exposed REST and gRPC inference endpoints.
This guide assumes you already have a functional Kubernetes cluster and helm 
installed (see below for instructions on installing helm).

The steps below describe how to setup a model repository, use helm to launch the inference server and then send 
inference requests to the running server. 

## Installing Helm

Please refer to [Helm installation guide](https://helm.sh/docs/intro/install).

## Model Repository

Model Server requires a repository of models to execute inference requests. That consists of the model files stored in a 
specific structure. Each model is stored in a dedicated folder with numerical subfolders representing the model versions.
Each model version subfolder must include its model files. 

Model repository can be hosted in the cloud storage, Kubernetes persistent volume or on the local drives.

Learn more about the [model repository](../docs/models_repository.md).

For example, you can 
use a Google Cloud Storage (GCS) bucket:
```shell script
gsutil mb gs://model-repository
```

You can download the model from [OpenVINO Model Zoo](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin). and upload it to GCS:

```shell script
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.bin -P 1
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.3/models_bin/2/resnet50-binary-0001/FP32-INT1/resnet50-binary-0001.xml -P 1
gsutil cp -r 1 resnet50-binary-0001.bin gs://model-repository/resnet
```

The supported storage options are described below:


### GCS

Bucket permissions can be set with the _GOOGLE_APPLICATION_CREDENTIALS_ environment variable. Please follow the steps below:

* Generate Google service account JSON file with permissions: _Storage Legacy Bucket Reader_, _Storage Legacy Object Reader_, _Storage Object Viewer_. Name a file for example: _gcp-creds.json_ 
(you can follow these instructions to [create a Service Account](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account) and download JSON)
* Create a Kubernetes secret from this JSON file:

      $ kubectl create secret generic gcpcreds --from-file gcp-creds.json

* When deploying Model Server, provide the model path to GCS bucket and name for the secret created above. Make sure to provide `gcp_creds_secret_name` when deploying:
```shell script
helm install ovms-app ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository/model,gcp_creds_secret_name=gcpcreds
```

### S3

For S3 you must provide an AWS Access Key ID, the content of that key (AWS Secret Access Key) and the AWS region when deploying: `aws_access_key_id`, `aws_secret_access_key` and `aws_region` (see below).
```shell script
helm install ovms-app ovms --set model_name=icnet-camvid-ava-0001,model_path=s3://models-repository/model,aws_access_key_id=<...>,aws_secret_access_key=<...>,aws_region=eu-central-1
```

In case you would like to use custom S3 service with compatible API (e.g. MinIO), you need to also provide endpoint 
to that service. Please provide it by supplying `s3_compat_api_endpoint`:
```shell script
helm install ovms-app ovms --set model_name=icnet-camvid-ava-0001,model_path=s3://models-repository/model,aws_access_key_id=<...>,aws_secret_access_key=<...>,s3_compat_api_endpoint=<...>
```

### Azure Storage
Use OVMS with models stored on azure blob storage by providing `azure_storage_connection_string` parameter. Model path should follow `az` scheme like below:
```shell script
helm install ovms-app ovms --set model_name=resnet,model_path=az://bucket/model_path,azure_storage_connection_string="DefaultEndpointsProtocol=https;AccountName=azure_account_name;AccountKey=smp/hashkey==;EndpointSuffix=core.windows.net"
```
 
### Local Node Storage
Beside the cloud storage, models could be stored locally on the kubernetes nodes filesystem.
Use the parameter `models_host_path` with the local path on the nodes. It will be mounted in the OVMS container as `/models` folder.

While the models folder is mounted in the OVMS container, the parameter `model_path` should refer to the path starting with /models/... and point to the folder with the model versions.

Note that the OVMS container starts, by default, with the security context of account `ovms` with pid 5000 and group 5000. 
If the mounted models have restricted access permissions, change the security context of the OVMS service or adjust permissions to the models. OVMS requires read permissions on the model files and 
list permission on the model version folders.

### Persistent Volume
It is possible to deploy OVMS using Kubernetes [persistent volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/).

That opens a possibility of storing the models for OVMS on all Kubernetes [supported filesystems](https://kubernetes.io/docs/concepts/storage/storage-classes/).

In the helm set the parameter `models_volume_claim` with the name of the `PersistentVolumeClaim` record with the models. While set, it will be mounted as `/models` folder inside the OVMS container.

Note that parameter `models_volume_claim` is mutually exclusive with `models_host_path`. Only one of them should be set.

## Assigning Resource Specs

You can restrict assigned cluster resources to the OVMS container by setting the parameter `resources`.
By default, there are no restrictions but that parameter could be used to reduce the CPU and memory allocation. Below is the snippet example from the [values.yaml](https://github.com/openvinotoolkit/model_server/blob/develop/deploy/ovms/values.yaml) file:
```yaml
resources:
  limits:
    cpu: 8.0
    memory: 512Mi
```
Beside setting the CPU and memory resources, the same parameter can be used to assign AI accelerators like iGPU, or VPU.
That assumes using adequate Kubernetes device plugin from [Intel Device Plugin for Kubernetes](https://github.com/intel/intel-device-plugins-for-kubernetes).
```yaml
resources:
  limits:
    gpu.intel.com/i915: 1
```

## Security Context

OVMS, by default, starts with the security context of `ovms` account which has pid 5000 and gid 5000. In some cases it can prevent importing models
stored on the file system with restricted access.
It might require adjusting the security context of OVMS service. It can be changed using a parameter `security_context`.

An example of the values is presented below:
```yaml
security_context:
  runAsUser: 5000
  runAsGroup: 5000
``` 
The security configuration could be also adjusted further with all options specified in [Kubernetes documentation](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) 

## Service Type
The helm chart creates the Kubernetes `service` as part of the OVMS deployment. Depending on the cluster infrastructure you can adjust
the [service type](https://kubernetes.io/docs/concepts/services-networking/service/#publishing-services-service-types).
In the cloud environment you might set `LoadBalancer` type to expose the service externally. `NodePort` could expose a static port
of the node IP address. `ClusterIP` would keep the OVMS service internal to the cluster applications.  

    
## Deploy OpenVINO Model Server with a Single Model

Deploy Model Server using _helm_. Please include the required model name and model path. You can also adjust other parameters defined in [values.yaml](https://github.com/openvinotoolkit/model_server/tree/develop/deploy/ovms/values.yaml).
```shell script
helm install ovms-app ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository
```

Use _kubectl_ to see the status and wait until the Model Server pod is running:
```shell script
kubectl get pods
NAME                    READY   STATUS    RESTARTS   AGE
ovms-app-5fd8d6b845-w87jl   1/1     Running   0          27s
```

By default, Model Server is deployed with 1 instance. If you would like to scale up additional replicas, override the value in values.yaml file or by passing _--set_ flag to _helm install_:

```shell script
helm install ovms-app ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository,replicas=3
```


## Deploy OpenVINO Model Server with Multiple Models Defined in a Configuration File

To serve multiple models you can run Model Server with a configuration file as described in [Config File](../docs/multiple_models_mode.md).

Follow the above documentation to create a configuration file named _config.json_ and fill it with proper information.

To deploy with config file stored in the Kubernetes ConfigMap:
* create a ConfigMap resource from this file with a chosen name (here _ovms-config_):
```shell script     
kubectl create configmap ovms-config --from-file config.json
```
* deploy Model Server with parameter `config_configmap_name` (without `model_name` and `model_path`):
```shell script
helm install ovms-app ovms --set config_configmap_name=ovms-config
```
To deploy with config file stored on the Kubernetes Persistent Volume :
* Store the config file on node local path set with `models_host_path` or on the persistent volume claim set with 
`models_claim_name`. It will be mounted along with the models in the folder `/models`.
* Deploy Model Server with parameter `config_path` pointing to the location of the config file visible in the OVMS container ie starting from 
`/models/...`
```shell script
helm install ovms-app ovms --set config_path=/models/config.json
```
## Using Model Server

Now that the server is running you can send HTTP or gRPC requests to perform inference. 
By default, the service is exposed with a LoadBalancer service type. Use the following command to find the 
external IP for the server:
```shell script
kubectl get svc
NAME                    TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                         AGE
ovms-app   LoadBalancer   10.121.14.253   1.2.3.4         8080:30043/TCP,8081:32606/TCP   59m
```

The server exposes an gRPC endpoint on 8080 port and REST endpoint on 8081 port.

The service name deployed via the helm chart is defined by the application name. In addition to that, the service
gets a suffix `-ovms`, in case the application name doesn't include `ovms` phrase. It avoids a risk of the service name
conflicts with other application.

Follow the [instructions](https://github.com/openvinotoolkit/model_server/tree/develop/demos/image_classification/python) 
to create an image classification client that can be used to perform inference with models being exposed by the server. For example:
```shell script
$ python image_classification.py --grpc_port 8080 --grpc_address 1.2.3.4 --input_name 0 --output_name 1463
Start processing:
	Model name: resnet
	Images list file: input_images.txt
images/airliner.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 25.56 ms; speed 39.13 fps
Detected: 404  Should be: 404
images/arctic-fox.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 20.95 ms; speed 47.72 fps
Detected: 279  Should be: 279
images/bee.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.90 ms; speed 45.67 fps
Detected: 309  Should be: 309
images/golden_retriever.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.84 ms; speed 45.78 fps
Detected: 207  Should be: 207
images/gorilla.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 20.26 ms; speed 49.36 fps
Detected: 366  Should be: 366
images/magnetic_compass.jpeg (1, 3, 224, 224) ; data range: 0.0 : 247.0
Processing time: 20.68 ms; speed 48.36 fps
Detected: 635  Should be: 635
images/peacock.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.57 ms; speed 46.37 fps
Detected: 84  Should be: 84
images/pelican.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 20.53 ms; speed 48.71 fps
Detected: 144  Should be: 144
images/snail.jpeg (1, 3, 224, 224) ; data range: 0.0 : 248.0
Processing time: 22.34 ms; speed 44.75 fps
Detected: 113  Should be: 113
images/zebra.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 21.27 ms; speed 47.00 fps
Detected: 340  Should be: 340
Overall accuracy= 100.0 %
Average latency= 21.1 ms
```

## Cleanup

Once you've finished using the server you should use helm to uninstall the chart:
```shell script
$ helm ls
NAME  	  NAMESPACE	REVISION	UPDATED                                 	STATUS  	CHART     	APP VERSION
ovms-app  default  	1       	2020-09-23 14:40:07.292360971 +0200 CEST	deployed	ovms-3.0.0

$ helm uninstall ovms-app
release "ovms-app" uninstalled
```


## Helm Options References

| Parameter        | Description           | Prerequisites  | Default |
| ------------- |-------------|-------------|-------------|
| replicas      | Number of k8s pod replicas to deploy  | - | 1 |
| image_name      | Change to use different docker image with OVMS   | - | openvino/model_server:latest |
| config_configmap_name | Starts OVMS using the config file stored in the ConfigMap |    Create the ConfigMap including config.json file | - |
| config_path | Starts OVMS using the config file mounted from the node local path or the k8s persistent volume | Use it together with models_host_path or models_claim_name and place the config file in configured storage path | - |
| grpc_port      | Service port for gRPC interface | - | 8080 |
| grpc_port      | Service port for REST API interface | - | 8081 |
| file_system_poll_wait_seconds      | Time interval in seconds between new version detection. 0 disables the version updates | - | 1 |
| model_name      | Model name, start OVMS with a single model, excluding with config_configmap_name and config_path parameter | - | - |
| model_path      | Model path, start OVMS with a single model, excluding with config_configmap_name and config_path parameter | - | - |
| target_device      | Target device to run inference operations | Non CPU device require the device plugin to be deployed | CPU |
| stateful      | If set to any non empty value, enables stateful model execution | Model must be stateful | Stateless model execution |
| nireq      | Size of inference queue  | - | Set automatically by OpenVINO|
| batch_size      | Change the model batch size  | - | Defined by the model attributes |
| layout      | Change layout of the model input or output with image data. NCHW or NHWC  | - | Defined in the model |
| shape      | Change the model input shape  | - | defined by the model attributes |
| model_version_policy      | Set the model version policy  | - | {\"latest\": { \"num_versions\":1 }} The latest version is served |
| plugin_config      | Device plugin configuration used for performance tuning  | - | {\"CPU_THROUGHPUT_STREAMS\":\"CPU_THROUGHPUT_AUTO\"}|
| gcp_creds_secret_name      | k8s secret resource including GCP credentials, use it with google storage for models | Secret should be created with GCP credentials json file | - |
| aws_access_key_id      | S3 storage access key id, use it with S3 storage for models | - | - |
| aws_secret_access_key      | S3 storage secret key, use it with S3 storage for models | - | - |
| aws_region      | S3 storage secret key, use it with S3 storage for models | - | - |
| aws_secret_access_key      | S3 storage secret key, use it with S3 storage for models |-  | - |
| s3_compat_api_endpoint      | S3 compatibility api endpoint, use it with Minio storage for models |  | - |
| azure_storage_connection_string   | Connection string to the Azure Storage authentication account, use it with Azure storage for models | - | - |
| log_level      | OVMS log level, one of ERROR,INFO,DEBUG| - |  INFO |
| service_type      | k8s service type | - | LoadBalancer |
| resources      | Compute resource limits | - | All CPU and memory on the node |
| node_selector      | Target node label condition | - | All available nodes |
| annotations      | Defined annotations to be set in the pods | - | None |
| security_context     | OVMS security context | - | 5000:5000 |
| models_host_path      | Mounts node local path in container as /models folder | Path should be created on all nodes and populated with the data | - |
| models_volume_claim      | Mounts k8s persistent volume claim in the container as /models | Persistent Volume Claim should be create in the same namespace and populated with the data | - |
| https_proxy | Proxy name to be used to connect to remote models | - | - |
