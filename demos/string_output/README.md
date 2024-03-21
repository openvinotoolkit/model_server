# String output model demo {#ovms_string_output_model_demo}
## Overview

This demo demonstrates usage of OVMS with model that returns string as an output.

### Download Image Net modl

```bash
python3 download_model.py
```

### Start the OVMS container:
```bash
docker run -d -u $(id -u):$(id -g) -v $(pwd):/workspace -p 9178:9178 openvino/model_server:latest \
--model_path /model --model_name text --rest_port 9178
```

## Send request
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

## Expected output
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