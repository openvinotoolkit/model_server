# Installation of OpenVINO&trade; Model Server with Kubernetes and Helms Chart

A helm chart for installing OpenVINO Model Server in a Kubernetes cluster is provided. By default, the cluster contains a single instance of the server but the _replicas_ configuration parameter can be set to create a cluster of any size, as described below. This guide assumes you already have a functional Kubernetes cluster and helm installed (see below for instructions on installing helm).

This guide describes how to setup a model repository, use helm to launch the inference server and then send inference requests to the running server. 

## Installation of Helm

Refer this [link](https://helm.sh/docs/intro/install) for Helms Installation
## Setting up the Model Repository

If you already have a model repository you may use that with this helm chart. If you don't, you can use any model from the [OpenVino Model Zoo](https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/).
 
Model Server requires a repository of models to execute inference requests. For example, you can 
use a Google Cloud Storage (GCS) bucket:
```shell script
$ gsutil mb gs://model-repository
```

You can download the model from the link provided above and upload it to GCS:
```shell script
$ gsutil cp -r 1 gs://model-repository/1
```

## Bucket Permissions for Cloud Storage

Make sure to set bucket permissions so the server can access the model repository. If the bucket 
is public or Model Server is run on the same Google Cloud Platform (GCP) or Amazon Web Services (AWS) account as the storage bucket, then no additional changes 
are needed and you can proceed to _Deploy the Model Server_ section.

### Google Cloud Storage.

Bucket permissions can be set with the _GOOGLE_APPLICATION_CREDENTIALS_ environment variable. Please follow the steps below:

* Generate Google service account JSON file with permissions: _Storage Legacy Bucket Reader_, _Storage Legacy Object Reader_, _Storage Object Viewer_. Name a file for example: _gcp-creds.json_ 
(you can follow these instructions to create a Service Account and download JSON [here](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account) )
* Create a Kubernetes secret from this JSON file:

``` bash 

      $ kubectl create secret generic gcpcreds --from-file gcp-creds.json
```

* When deploying Model Server, provide the model path to GCS bucket and name for the secret created above. Make sure to provide `gcp_creds_secret_name` when deploying:
```shell script
$ helm install ovms ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository,gcp_creds_secret_name=gcpcreds
```

### Amazon Web Services - Amazon S3

For S3 you must provide an AWS Access Key ID, the content of that key (AWS Secret Access Key) and the AWS region when deploying: `aws_access_key_id`, `aws_secret_access_key` and `aws_region`
```shell script
$ helm install ovms ovms --set model_name=icnet-camvid-ava-0001,model_path=s3://models-repository,aws_access_key_id=<...>,aws_secret_access_key=<...>,aws_region=eu-central-1
```

In case you would like to use custom S3 service with compatible API (e.g. MinIO), you need to also provide endpoint 
to that service. Please provide it by supplying `s3_compat_api_endpoint`:
```shell script
$ helm install ovms ovms --set model_name=icnet-camvid-ava-0001,model_path=s3://models-repository,aws_access_key_id=<...>,aws_secret_access_key=<...>,s3_compat_api_endpoint=<...>
```
    
## Deploy the Model Server Image

Deploy Model Server using _helm_. Please include the required model name and model path. You can also adjust other parameters defined in [values.yaml](../deploy/ovms/values.yaml)
$ helm install ovms ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository
```

Use _kubectl_ to see status and wait until the Model Server pod is running:
```shell script
$ kubectl get pods
NAME                    READY   STATUS    RESTARTS   AGE
ovms-5fd8d6b845-w87jl   1/1     Running   0          27s
```

By default, Model Server is deployed with 1 instance. If you would like to scale up additional replicas, override the value in values.yaml file or by passing _--set_ flag to _helm install_:

```shell script
$ helm install ovms ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository,replicas=3
```
## Deploy Model Server with a Configuration File

To serve multiple models you can run Model Server with a configuration file as described [here](./InstallationsLinuxDocker.md#configfile)

### To deploy with config file:
- Create a configuration file named config.json and fill it with proper information

- Create a configmap from this file with a chosen name (here ovms-config):

```bash
kubectl create configmap ovms-config --from-file config.json
deploy Model Server with parameter config_configmap_name (without model_name and model_path):

helm install ovms ovms --set config_configmap_name=ovms-config

```

## Using the Model Server

Now that the server is running you can send HTTP or gRPC requests to get inference. 
By default, the service is exposed with a LoadBalancer service type. Use the following command to find the 
external IP for the server:
```shell script
$ kubectl get svc
NAME                    TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                         AGE
openvino-model-server   LoadBalancer   10.121.14.253   1.2.3.4         8080:30043/TCP,8081:32606/TCP   59m
```

The server exposes an gRPC endpoint on 8080 port and REST endpoint on 8081 port.

Follow the instructions [here](https://github.com/openvinotoolkit/model_server/tree/main/example_client#submitting-grpc-requests-based-on-a-dataset-from-a-list-of-jpeg-files)
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

## Cleanup of Resources

Once you've finished using the server you should use helm to uninstall the chart:
```shell script
$ helm ls
NAME	NAMESPACE	REVISION	UPDATED                                 	STATUS  	CHART     	APP VERSION
ovms	default  	1       	2020-09-23 14:40:07.292360971 +0200 CEST	deployed	ovms-2.0.0

$ helm uninstall ovms
release "ovms" uninstalled
```

You may also want to delete the GCS bucket you created to hold the model repository:

GCS:
```shell script
$ gsutil rm -r gs://model-repository
```

S3:
```shell script
$ aws s3 rb s3://models-repository
```
