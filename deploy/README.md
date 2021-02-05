# Kubernetes Deployment

A helm chart for installing OpenVINO Model Server in a Kubernetes cluster is provided. By default, the cluster contains 
a single instance of the server but the _replicas_ configuration parameter can be set to create a cluster 
of any size, as described below. This guide assumes you already have a functional Kubernetes cluster and helm 
installed (see below for instructions on installing helm).

The steps below describe how to setup a model repository, use helm to launch the inference server and then send 
inference requests to the running server. 

## Installing Helm

Please refer to: https://helm.sh/docs/intro/install for Helm installation.

## Model Repository

If you already have a model repository you may use that with this helm chart. If you don't, you can use any model
from https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/.

Model Server requires a repository of models to execute inference requests. For example, you can 
use a Google Cloud Storage (GCS) bucket:
```shell script
$ gsutil mb gs://model-repository
```

You can download the model from the link provided above and upload it to GCS:
```shell script
$ gsutil cp -r 1 gs://model-repository/1
```
The models repository can be also distributed on the cluster nodes in the local path or it could be stored on the Kubernetes persistent volume.
Below are described supported storage options:


### GCS

Bucket permissions can be set with the _GOOGLE_APPLICATION_CREDENTIALS_ environment variable. Please follow the steps below:

* Generate Google service account JSON file with permissions: _Storage Legacy Bucket Reader_, _Storage Legacy Object Reader_, _Storage Object Viewer_. Name a file for example: _gcp-creds.json_ 
(you can follow these instructions to create a Service Account and download JSON: 
https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
* Create a Kubernetes secret from this JSON file:

      $ kubectl create secret generic gcpcreds --from-file gcp-creds.json

* When deploying Model Server, provide the model path to GCS bucket and name for the secret created above. Make sure to provide `gcp_creds_secret_name` when deploying:
```shell script
$ helm install ovms-app ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository,gcp_creds_secret_name=gcpcreds
```

### S3

For S3 you must provide an AWS Access Key ID, the content of that key (AWS Secret Access Key) and the AWS region when deploying: `aws_access_key_id`, `aws_secret_access_key` and `aws_region` (see below)
```shell script
$ helm install ovms-app ovms --set model_name=icnet-camvid-ava-0001,model_path=s3://models-repository,aws_access_key_id=<...>,aws_secret_access_key=<...>,aws_region=eu-central-1
```

In case you would like to use custom S3 service with compatible API (e.g. MinIO), you need to also provide endpoint 
to that service. Please provide it by supplying `s3_compat_api_endpoint`:
```shell script
$ helm install ovms-app ovms --set model_name=icnet-camvid-ava-0001,model_path=s3://models-repository,aws_access_key_id=<...>,aws_secret_access_key=<...>,s3_compat_api_endpoint=<...>
```

### Azure Storage
Use OVMS with models stored on azure blob storage by providing `azure_storage_connection_string` parameter. Model path should follow `az` scheme like below:
```shell script
$ helm install ovms-app ovms --set model_name=resnet,model_path=az://bucket/model_path,azure_storage_connection_string="DefaultEndpointsProtocol=https;AccountName=azure_account_name;AccountKey=smp/hashkey==;EndpointSuffix=core.windows.net"
```
 
### Local Node Storage
Beside the cloud storage, models could be stored locally on the kubernetes nodes filesystem.
Use the parameter `models_host_path` with the local path on the nodes. It will be mounted in the OVMS container as `/models` folder.

While the models folder is mounted in the OVMS container, the parameter `model_path` should refer to the path staring with /models/... and point to the folder with the model versions.

Note that the OVMS container starts, by default, with the security context of account `ovms` with pid 5000 ahd group 5000. 
If the mounted models have restricted access permissions, change the security context of the OVMS service or adjust permissions to the models. OVMS requires read permissions on the model files and 
list permission on the model version folders.

### Persistent Volume
If is possible to deploy OVMS using Kubernetes [persistent volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/).

That opens a possibility of storing the using the models for OVMS on all Kubernetes [supported filesystems](https://kubernetes.io/docs/concepts/storage/storage-classes/).

In the helm set the parameter `models_volume_claim` with the name of the `PersistentVolumeClaim` record with the models. While set, it will be mounted as `/models` folder inside the OVMS container.

Note that parameter `models_volume_claim` is excluding with `models_host_path`. Only one of them should be set.

## Assigning Resource Specs

You can restrict assigned cluster resources to the OVMS container but setting the parameter `resources`.
Be default, there are no restrictions but it could be used to reduce the CPU and memory like in the snippet example from the values.yaml file:
```yaml
resources:
  limits:
    cpu: 8.0
    memory: 512Mi
```
Beside setting the CPU and memory resources, the same parameter can be used to assign AI accelerators like iGPU, or VPU.
That assume using adequate Kubernetes device plugin from [Intel Device Plugin for Kubernetes](https://github.com/intel/intel-device-plugins-for-kubernetes).
```yaml
resources:
  limits:
    gpu.intel.com/i915: 1
```

## Security Context

OVMS, by default, starts with the security context of `ovms` account which has pid 5000 and gid 5000. In some cases it can prevent accessing models
stored on the file system with restricted access.
It might require adjusting the security context of OVMS service. It can be changed using a parameter `security_context`.

Example of the values is below:
```yaml
security_context:
  runAsUser: 5000
  runAsGroup: 5000
``` 
The configuration could be also adjusted like allowed according to [Kubernetes documentation](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/) 

## Service Type
The heml chart creates the Kubernetes `service` as part of the OVMS deployment. Depending on the cluster infrastructure you can adjust
the [service type](https://kubernetes.io/docs/concepts/services-networking/service/#publishing-services-service-types).
In the cloud environment you might set `LoadBalancer` type to expose the service externally. `NodePort` could expose a static port
of the node IP address. `ClusterIP` would keep the OVMS service internal to the cluster applications.  

    
## Deploy OpenVINO Model Server with a Single Model

Deploy Model Server using _helm_. Please include the required model name and model path. You can also adjust other parameters defined in [values.yaml](ovms/values.yaml).
```shell script
$ helm install ovms-app ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository
```

Use _kubectl_ to see status and wait until the Model Server pod is running:
```shell script
$ kubectl get pods
NAME                    READY   STATUS    RESTARTS   AGE
ovms-5fd8d6b845-w87jl   1/1     Running   0          27s
```

By default, Model Server is deployed with 1 instance. If you would like to scale up additional replicas, override the value in values.yaml file or by passing _--set_ flag to _helm install_:

```shell script
$ helm install ovms-app ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository,replicas=3
```


## Deploy OpenVINO Model Server with Multiple Models Defined in a Configuration File

To serve multiple models you can run Model Server with a configuration file as described here:
https://github.com/openvinotoolkit/model_server/blob/main/docs/docker_container.md#configfile

Follow the above documentation to create a configuration file named _config.json_ and fill it with proper information.

To deploy with config file stored in the Kubernetes ConfigMap:
* create a ConfigMap resource from this file with a chosen name (here _ovms-config_):
      
      $ kubectl create configmap ovms-config --from-file config.json

* deploy Model Server with parameter `config_configmap_name` (without `model_name` and `model_path`):

      $ helm install ovms-app ovms --set config_configmap_name=ovms-config

To deploy with config file stored on the Kubernetes Persistent Volume :
* Store the config file on node local path set with `models_host_path` or on the persistent volume claim set with 
`models_claim_name`. It will be mounted along with the models in the folder `/models`.
* Deploy Model Server with parameter `config_path` pointing to the location of the config file visible in the OVMS container ie starting from 
`/models/...`
      $ helm install ovms-app ovms --set config_path=/models/config.json

## Using Model Server

Now that the server is running you can send HTTP or gRPC requests to perform inference. 
By default, the service is exposed with a LoadBalancer service type. Use the following command to find the 
external IP for the server:
```shell script
$ kubectl get svc
NAME                    TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                         AGE
openvino-model-server   LoadBalancer   10.121.14.253   1.2.3.4         8080:30043/TCP,8081:32606/TCP   59m
```

The server exposes an gRPC endpoint on 8080 port and REST endpoint on 8081 port.

Follow the instructions here: https://github.com/openvinotoolkit/model_server/tree/master/example_client#submitting-grpc-requests-based-on-a-dataset-from-a-list-of-jpeg-files 
to create an image classification client that can be used to perform inference with models being exposed by the server. For example:
```shell script
$ python jpeg_classification.py --grpc_port 8080 --grpc_address 1.2.3.4 --input_name data --output_name prob
	Model name: resnet
	Images list file: input_images.txt

images/airliner.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 73.00 ms; speed 2.00 fps 13.79
Detected: 895  Should be: 404
images/arctic-fox.jpeg (1, 3, 224, 224) ; data range: 7.0 : 255.0
Processing time: 52.00 ms; speed 2.00 fps 19.06
Detected: 279  Should be: 279
images/bee.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 82.00 ms; speed 2.00 fps 12.2
Detected: 309  Should be: 309
images/golden_retriever.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 86.00 ms; speed 2.00 fps 11.69
Detected: 207  Should be: 207
images/gorilla.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 65.00 ms; speed 2.00 fps 15.39
Detected: 366  Should be: 366
images/magnetic_compass.jpeg (1, 3, 224, 224) ; data range: 0.0 : 247.0
Processing time: 51.00 ms; speed 2.00 fps 19.7
Detected: 635  Should be: 635
images/peacock.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 61.00 ms; speed 2.00 fps 16.28
Detected: 84  Should be: 84
images/pelican.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 61.00 ms; speed 2.00 fps 16.41
Detected: 144  Should be: 144
images/snail.jpeg (1, 3, 224, 224) ; data range: 0.0 : 248.0
Processing time: 56.00 ms; speed 2.00 fps 17.74
Detected: 113  Should be: 113
images/zebra.jpeg (1, 3, 224, 224) ; data range: 0.0 : 255.0
Processing time: 73.00 ms; speed 2.00 fps 13.68
Detected: 340  Should be: 340

Overall accuracy= 90.0
```

## Cleanup

Once you've finished using the server you should use helm to uninstall the chart:
```shell script
$ helm ls
NAME  	  NAMESPACE	REVISION	UPDATED                                 	STATUS  	CHART     	APP VERSION
ovms-app  default  	1       	2020-09-23 14:40:07.292360971 +0200 CEST	deployed	ovms-3.0.0

$ helm uninstall ovms
release "ovms" uninstalled
```


## Helm Options References

| Parameter        | Description           | Prerequisites  | Default |
| ------------- |-------------|-------------|-------------|
| replicas      | number of k8s pod replicas to deploy  |  | 1 |
| image_name      | change to use different docker image with OVMS   |  | openvino/model_server:latest |
| config_configmap_name | Starts OVMS using the config file stored in the ConfigMap |    Create the ConfigMap including config.json file | - |
| config_path | Starts OVMS using the config file mounted from the node local path or the k8s persistent volume | Use it together with models_host_path or models_claim_name and place the config file in configured storage path | - |
| grpc_port      | service port for gRPC interface |  | 8080 |
| grpc_port      | service port for REST API interface |  | 8081 |
| model_name      | model name, start OVMS with a single model, excluding with config_configmap_name and config_path parameter |  | - |
| model_path      | model path, start OVMS with a single model, excluding with config_configmap_name and config_path parameter |  | - |
| target_device      | Target device to run inference operations | Non CPU device require the device plugin to be deployed | CPU |
| nireq      | Size of inference queue  |  | set automatically by OpenVINO|
| plugin_config      | Device plugin configuration used for performance tuning  |  | {\"CPU_THROUGHPUT_STREAMS\":\"CPU_THROUGHPUT_AUTO\"}|
| gcp_creds_secret_name      | k8s secret resource including GCP credentials, use it with google storage for models | Secret should be created with GCP credentials json file | - |
| aws_access_key_id      | S3 storage access key id, use it with S3 storage for models |  | - |
| aws_secret_access_key      | S3 storage secret key, use it with S3 storage for models |  | - |
| aws_region      | S3 storage secret key, use it with S3 storage for models |  | - |
| aws_secret_access_key      | S3 storage secret key, use it with S3 storage for models |  | - |
| s3_compat_api_endpoint      | S3 compatibility api endpoint, use it with Minio storage for models |  | - |
| azure_storage_connection_string   | S3 compatibility api endpoint, use it with Minio storage for models |  | - |
| log_level      | OVMS log level, one of ERROR,INFO,DEBUG|  |  INFO |
| service_type      | k8s service type |  | LoadBalancer |
| resources      | compute resource limits |  | All CPU and memory on the node |
| security_context     | OVMS security context |  | 5000:5000 |
| models_host_path      | mounts node local path in container as /models folder | Path should be created on all nodes and populated with the data | - |
| models_volume_claim      | mounts k8s persistent volume claim in the container as /models | Persistent Volume Claim should be create in the same namespace and populated with the data | - |
| https_proxy | proxy name to be used to connect to remote models | | - |

