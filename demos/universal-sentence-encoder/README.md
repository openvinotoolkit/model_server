# Using inputs data in string format with universal-sentence-encoder model {#ovms_demo_universal-sentence-encoder}


## Download the model

In this experiment we are going to use a TensorFlow model from [Kaggle](https://www.kaggle.com/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2).

```console
curl --create-dirs -L -o universal-sentence-encoder-multilingual/1/3.tar.gz https://www.kaggle.com/api/v1/models/google/universal-sentence-encoder/tensorFlow2/multilingual/2/download
tar -xzf universal-sentence-encoder-multilingual/1/3.tar.gz -C universal-sentence-encoder-multilingual/1/
rm universal-sentence-encoder-multilingual/1/3.tar.gz
```

Make sure the downloaded model has right permissions
```bash
chmod -R 755 universal-sentence-encoder-multilingual
```

The model setup should look like this
```bash
tree universal-sentence-encoder-multilingual/

universal-sentence-encoder-multilingual/
└── 1
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index

```

## Use OpenVINO tokenizers library

Model universal-sentence-encoder-multilingual includes a layer SentencepieceTokenizer which is supported via [OpenVINO custom extension](https://github.com/openvinotoolkit/openvino_tokenizers). It is dynamic library performing the execution of the model layer, it extends original set of supported OpenVINO operations.

The image `openvino/model_server:2023.3` and newer includes ready to use OpenVINO Model Server with the CPU extension.

## Start the model server in a container
You can start the service with a command:
```bash
docker run -d --name ovms -p 9000:9000 -p 8000:8000 -v $(pwd)/universal-sentence-encoder-multilingual:/model openvino/model_server:latest --model_name usem --model_path /model --cpu_extension /ovms/lib/libopenvino_tokenizers.so --plugin_config "{\"NUM_STREAMS\": 1}" --port 9000 --rest_port 8000
```

Check the container logs to confirm successful start:
```bash
docker logs ovms
```

Alternatively see (instructions)[https://github.com/openvinotoolkit/model_server/blob/main/docs/deploying_server_baremetal.md] for deployment on bare metal.

Make sure to:

- **On Windows**: run `setupvars` script
- **On Linux**: set `LD_LIBRARY_PATH` and `PATH` environment variables

on every shell that will start OpenVINO Model Server.

And start Model Server using the following command:
```bat
ovms --model_name usem --model_path universal-sentence-encoder-multilingual/ --plugin_config "{\"NUM_STREAMS\": 1}" --port 9000 --rest_port 8000
```

## Send string data as inference request

OpenVINO Model Server can accept the input in a form of strings. Below is a code snipped based on `tensorflow_serving_api` python library:
```python
data = np.array(["string1", "string1", "string_n"])
predict_request = predict_pb2.PredictRequest()
predict_request.model_spec.name = "my_model"
predict_request.inputs["input_name"].CopyFrom(make_tensor_proto(data))
predict_response = prediction_service_stub.Predict(predict_request, 10.0)
```

Clone the repo:
```console
git clone https://github.com/openvinotoolkit/model_server
```

Here is a basic client execution:
```console
pip install --upgrade pip
pip install -r model_server/demos/universal-sentence-encoder/requirements.txt
python model_server/demos/universal-sentence-encoder/send_strings.py --grpc_port 9000 --string "I enjoy taking long walks along the beach with my dog."
processing time 6.931 ms.
Output shape (1, 512)
Output subset [-0.00552395  0.00599533 -0.01480555  0.01098945 -0.09355522 -0.08445048
 -0.02802683 -0.05219319 -0.0675998   0.03127321 -0.03223499 -0.01282092
  0.06131846  0.02626886 -0.00983501  0.00298059  0.00141201  0.03229365
  0.06957124  0.01543707]

```

The same can be achieved using REST API interface and even a simple `curl` command:

```bash
curl -X POST http://localhost:8000/v1/models/usem:predict \
-H 'Content-Type: application/json' \
-d '{"instances": ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]}'
```  


## Compare results with TFS

The same client code can be used to send the requests to TensorFlow Serving component. There is full compatibility in the API.

Start TFS container:
```bash
docker run -it -p 8500:8500 -p 9500:9500 -v $(pwd)/universal-sentence-encoder-multilingual:/models/usem -e MODEL_NAME=usem tensorflow/serving --port=9500 --rest_api_port=8500
```

Run the client
```bash
python model_server/demos/universal-sentence-encoder/send_strings.py --grpc_port 9500 --input_name inputs --output_name outputs --string "I enjoy taking long walks along the beach with my dog."

processing time 12.167000000000002 ms.
Output shape (1, 512)
Output subset [-0.00552387  0.00599531 -0.0148055   0.01098951 -0.09355522 -0.08445048
 -0.02802679 -0.05219323 -0.06759984  0.03127313 -0.03223493 -0.01282088
  0.06131843  0.02626882 -0.00983502  0.00298053  0.00141208  0.03229369
  0.06957125  0.01543701]

```

> NOTE: Do not use this model with `--cache_dir`, the model does not support caching.
