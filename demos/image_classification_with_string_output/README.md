# String output model demo {#ovms_string_output_model_demo}
## Overview

This demo demonstrates example deployment of a model with output precision `ov::element::string`. The output text is serialized into corresponding fields in gRPC proto/REST body. This allows the client to consume the text directly and avoid the process of label mapping or detokenization.

### Download and prepare Image Net model using Keras and TensorFlow

```bash
pip install -r requirements.txt
python3 download_model.py
```

### Start the OVMS container:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd):/workspace -p 9178:9178 openvino/model_server:latest \
--model_path /workspace/model --model_name image_net --rest_port 9178
```

## Send request
Example below presents how to send request using KServ API with binary data extension with output returned inside of JSON:
```bash
echo -n '{"inputs" : [{"name" : "image", "shape" : [1], "datatype" : "BYTES"}]}' > request.json
stat --format=%s request.json
70
printf "%x\n" `stat -c "%s" ../common/static/images/bee.jpeg`
1c21
echo -n -e '\x21\x1c\x00\x00' >> request.json
cat ../common/static/images/bee.jpeg >> request.json
curl --data-binary "@./request.json" -X POST http://localhost:9022/v2/models/image_net/versions/0/infer -H "Inference-Header-Content-Length: 70"
```
There is also a way to force OVMS to return output in binary data extension by adding binary_data parameter to the request:
```bash
echo -n '{"inputs" : [{"name" : "image", "shape" : [1], "datatype" : "BYTES"}], "outputs" : [{"name" : "label", "parameters" : {"binary_data" : true}}]}' > request.json
stat --format=%s request.json
143
printf "%x\n" `stat -c "%s" ../common/static/images/bee.jpeg`
1c21
echo -n -e '\x21\x1c\x00\x00' >> request.json
cat ../common/static/images/bee.jpeg >> request.json
curl --data-binary "@./request.json" -X POST http://localhost:9022/v2/models/image_net/versions/0/infer -H "Inference-Header-Content-Length: 143" --output response.json
```
Request may be sent also using other APIs (KServ GRPC, TFS). In this sections you can find short code samples how to do this:
- [TensorFlow Serving API](./clients_tfs.md)
- [KServe API](./clients_kfs.md)


## Expected output
With output inside of JSON:
```bash
{
    "model_name": "image_net",
    "model_version": "1",
    "outputs": [{
            "name": "label",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["bee"]
        }]
}
```
With output in binary extension:
```bash
{
    "model_name": "image_net",
    "model_version": "1",
    "outputs": [{
            "name": "label",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": {
                "binary_data_size": 7 }
        }]
}<0x03000000> bee
```
