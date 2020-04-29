# Kubernetes deployment

A helm chart for installing OpenVino Model Server on Kubernetes cluster is provided. By default the cluster contains 
a single instance of the server but the _replicas_ configuration parameter can be set to create a cluster 
of any size, as described below. This guide assumes you already have a functional Kubernetes cluster and helm 
installed (see below for instructions on installing helm).

The steps below describe how to setup a model repository, use helm to launch the inference server and then send 
inference requests to the running server. 

## Installing Helm

Please refer to: https://helm.sh/docs/intro/install for Helm installation.

## Model repository

If you already have a model repository you may use that with this helm chart. If you don't, you can use any model
from https://download.01.org/opencv/2020/openvinotoolkit/2020.2/open_model_zoo/models_bin .

OpenVINO Model Server needs models that it will make available for inferencing. For example you can 
use Google Cloud Storage bucket:
```shell script
gsutil mb gs://model-repository
```

You can download the model from the link provided above nad upload it to GCS:
```shell script
gsutil cp -r 1 gs://model-repository/1
```

## Bucket permissions

Make sure the bucket permissions are set so that the inference server can access the model repository. If the bucket 
is public then no additional changes are needed and you can proceed to _Using OpenVINO Model Server_ section.

If bucket permissions need to be set with the _GOOGLE_APPLICATION_CREDENTIALS_ environment variable then perform the 
following steps:

* Generate Google service account JSON with permissions: _Storage Legacy Bucket Reader_, _Storage Legacy Object Reader_,
 _Storage Object Viewer_ . Name a file for example: _gcp-creds.json_ 
(you can follow these instructions to create Service Account and download JSON: 
https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
* Create a Kubernetes secret from this file:

      $ kubectl create secret generic gcpcreds --from-file gcp-creds.json

* Modify templates/deployment.yaml to include the GOOGLE_APPLICATION_CREDENTIALS environment variable:
      
      env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /secret/gcp-creds.json

* Modify templates/deployment.yaml to mount the secret in a volume at /secret:

      volumeMounts:
        - name: gcpcreds
          mountPath: "/secret"
          readOnly: true
      ...
      volumes:
      - name: gcpcreds
        secret:
          secretName: gcpcreds
          
## Deploy the Model Server

Deploy the Model Server using _helm_ . Please provide also required model name and model path:
```shell script
helm install ovms ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository
```

Use _kubectl_ to see status and wait until the model server pod is running:
```shell script
$ kubectl get pods
NAME                    READY   STATUS    RESTARTS   AGE
ovms-5fd8d6b845-w87jl   1/1     Running   0          27s
```

By default the OVMS deploys with 1 instance. If you would like to scale it, you could override value in values.yaml
file or by passing _--set_ flag to _helm install_:

```shell script
helm install ovms ovms --set model_name=resnet50-binary-0001,model_path=gs://models-repository,replicas=3
```

## Using OpenVINO Model Server

Now that the server is running you can send HTTP or gRPC requests to it to perform inferencing. 
By default, the service is exposed with a LoadBalancer service type. Use the following to find the 
external IP for the server:
```shell script
$ kubectl get svc
NAME                    TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                         AGE
openvino-model-server   LoadBalancer   10.121.14.253   1.2.3.4         8080:30043/TCP,8081:32606/TCP   59m

```

The server exposes an gRPC endpoint on 8080 port and REST endpoint on 8081 port.

Follow the instructions here: https://github.com/IntelAI/OpenVINO-model-server/tree/master/example_client#submitting-grpc-requests-based-on-a-dataset-from-a-list-of-jpeg-files 
to get the example image classification client that can be used to perform inferencing using 
image classification models being served by the server. For example:

```shell script
python jpeg_classification.py --grpc_port 8080 --grpc_address 1.2.3.4 --input_name data --output_name prob
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
NAME    NAMESPACE       REVISION        UPDATED                                 STATUS          CHART           APP VERSION
ovms    default         1               2020-04-20 11:47:18.263135654 +0000 UTC deployed        ovms-v1.0.0

$ helm uninstall ovms
release "ovms" uninstalled
```

You may also want to delete the GCS bucket you created to hold the model repository:
```shell script
$ gsutil rm -r gs://model-repository
```
