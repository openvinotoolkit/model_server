# GPT-J demo

### Download the model

Prepare the environment:
```
pip install -r requirements.txt
```

Download the GPT-J-6b model from huggingface and save to disk in pytorch format:
```
python3 download_mode.py
```

### Convert the model
The model needs to be converted to ONNX format in order to load in OVMS:
```
./convert_model.sh
```
The model will reside in `onnx/1` directory.

There should be output with visible successful model validation results:

```
Validating ONNX model...
        -[✓] ONNX model output names match reference model ({'logits'})
        - Validating ONNX Model output "logits":
                -[✓] (3, 9, 50400) matches (3, 9, 50400)
                -[✓] all values close (atol: 0.0001)
All good, model saved at: onnx/1/model.onnx
```

### Start OVMS with GPT-J-6b model

```
docker run -it --rm -p 9000:9000 -v $(pwd)/onnx:/model:ro openvino/model_server \
    --port 9000 \
    --model_name gpt-j-6b \
    --model_path /model \
    --plugin_config '{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":1}' \
    --log_level DEBUG
```

### Run the OVMS client

```
python3 infer_ovms.py --url localhost:9000 --model_name gpt-j-6b
```

Output:
```
[[[ 8.407803   7.2024884  5.114844  ... -6.691438  -6.7890754 -6.6537027]
  [ 6.97011    9.89741    8.216569  ... -3.891536  -3.6937592 -3.6568289]
  [ 8.199201  10.721757   8.502647  ... -6.340912  -6.247861  -6.1362333]
  [ 6.5459595 10.398776  11.310042  ... -5.9843545 -5.806437  -6.0776973]
  [ 8.934336  13.137416   8.568134  ... -6.835008  -6.7942514 -6.6916494]
  [ 5.1626735  6.062623   1.7213026 ... -7.789153  -7.568969  -7.6591196]]]
predicted word:  a
```

### Run the inference with pytorch
We run the inference with pytorch to compare the result:
```
python3 infer_torch.py
```

Output:
```
tensor([[[ 8.4078,  7.2025,  5.1148,  ..., -6.6914, -6.7891, -6.6537],
         [ 6.9701,  9.8974,  8.2166,  ..., -3.8915, -3.6938, -3.6568],
         [ 8.1992, 10.7218,  8.5026,  ..., -6.3409, -6.2479, -6.1362],
         [ 6.5460, 10.3988, 11.3100,  ..., -5.9844, -5.8064, -6.0777],
         [ 8.9343, 13.1374,  8.5681,  ..., -6.8350, -6.7943, -6.6916],
         [ 5.1627,  6.0626,  1.7213,  ..., -7.7891, -7.5690, -7.6591]]],
       grad_fn=<ViewBackward0>)
predicted word:  a
```

### Interactive OVMS demo

Run `app.py` script to run interactive demo predicting the next word in a loop until end of sentence token is encountered.

```
python3 app.py --input "Neurons are fascinating"
```

Output:
```
Iteration: 61
Last predicted token: 198
Last latency: 0.9536168575286865s
Neurons are fascinating cells that are able to communicate with each other and with other cells in the body. Neurons are the cells that make up the nervous system, which is responsible for the control of all body functions. Neurons are also responsible for the transmission of information from one part of the body to another.
```
