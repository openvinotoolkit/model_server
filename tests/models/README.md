# Generating models with simple arithmetical operations

## Prepare Python environment

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/tests/models
pip3 install -r ../requirements.txt
```

## Model incrementing an input tensor

```bash
python increment.py
```

It generates saved_model format of TensorFlow Model adding "1" to every element o input tensor.
The model is stored in folder `/tmp/increment/1`.

The script is performing evaluation operation for a sample dataset.

Inspect the model format using command:

```bash
saved_model_cli show --dir /tmp/increment/1  --tag_set serve --signature_def serving_default
  inputs['in'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['out'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: output:0
```

Generate OpenVINO IR model format using model optimizer in a docker container:

```bash
docker run -it -v /tmp/increment/1:/model openvino/ubuntu20_dev mo --saved_model_dir /model/ --batch 1 --output_dir /model/

```

# Model calculating index with max value for the sum of two inputs

```bash
python argmax_sum.py --input_size 1000 --export_dir /tmp/argmax/1
```

```bash
saved_model_cli show --dir /tmp/argmax/  --tag_set serve --signature_def serving_default
  inputs['in1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1000)
      name: input1:0
  inputs['in2'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1000)
      name: input2:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['argmax'] tensor_info:
      dtype: DT_INT64
      shape: (-1)
      name: argmax:0
```


Generate OpenVINO IR model format using model optimizer in a docker container:

```bash
docker run -it -v /tmp/argmax:/model openvino/ubuntu20_dev mo --saved_model_dir /model/ --batch 1 --output_dir /model/

```
