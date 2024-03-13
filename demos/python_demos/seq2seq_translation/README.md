# Seq2seq demo with python node {#ovms_demo_python_seq2seq}

## Build image

From the root of the repository run:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

## Deploy OpenVINO Model Server with the Python calculator
Prerequisites:
-  image of OVMS with Python support and Optimum installed

Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python node

```bash
cd demos/python_demos/seq2seq_translation
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Requesting translation
Install client requirements

```bash
pip3 install -r requirements.txt 
```
Run the client script
```bash
python3 client.py --url localhost:9000
```

Expected output:
```bash
Text:
He never went out without a book under his arm, and he often came back with two.

Translation:
Il n'est jamais sorti sans un livre sous son bras, et il est souvent revenu avec deux.

```
