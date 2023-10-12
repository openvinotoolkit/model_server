# Llama 2 Chat  {#ovms_demo_llama_2_chat}

### Introduction
This demo showcases example usage of [Llama model](https://ai.meta.com/llama/) hosted via OpenVINO™ Model Server. The model used in this example can be found at [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf) (~26GB). Steps below automate download and conversion steps to be able to load it using OpenVINO™. Example python script is provided to request answers to given question.

### Download the model

Prepare the environment:
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/llama_chat/python
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the `meta-llama/Llama-2-7b-hf` model from huggingface and save to disk in IR format using script below.  
> NOTE: Download might take a while since the model is ~26GB.
```bash
python3 download_model.py
```
The model files should be available in models directory:
```bash
tree models

models
└── llama-2-7b-hf
    └── 1
        ├── config.json
        ├── openvino_model.bin
        └── openvino_model.xml
```

### Start OVMS with prepared Llama 2 model

```bash
docker run -d --rm -p 9000:9000 -v $(pwd)/models/llama-2-7b-hf:/model:ro openvino/model_server \
    --port 9000 \
    --model_name llama \
    --model_path /model \
    --plugin_config '{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":1}'
```

### Run python client

Run `client.py` script to run interactive demo. Available parameters:

```bash
python3 client.py -h

usage: client.py [-h] --url URL --question QUESTION [--seed SEED] [--actor {general-knowledge,python-programmer}]

Inference script for generating text with llama

optional arguments:
  -h, --help            show this help message and exit
  --url URL             Specify url to grpc service
  --question QUESTION   Question to selected actor
  --seed SEED           Seed for next token selection algorithm. Providing different numbers will produce slightly different results.
  --actor {general-knowledge,python-programmer}
                        Domain in which you want to interact with the model. Selects predefined pre-prompt.
```

Multiple examples for different pre-prompts (`--actor` parameter):  

General knowledge:
```bash
python3 client.py --url localhost:9000 --question "How many corners there are in square?" --seed 14140 --actor general-knowledge

 Four. [EOS]
```

Python programmer:
```bash
python3 client.py --url localhost:9000 --question "Write python function to sum 3 numbers." --seed 1332 --actor python-programmer

def sum_three_numbers(a,b,c):
   result = a + b + c
   return result [EOS]
```

>NOTE: You can edit the pre-prompt in `client.py` for your use case.
