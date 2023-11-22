## Run Model Server
Prerequisites:
-  image of OVMS with Python support and Optimum installed

Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and uses optimum-intel framework
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

```bash
docker run -it --rm -p 11339:11339 -v $PWD/servable:/workspace -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy openvino/model_server:py --config_path /workspace/config.json --log_level DEBUG --port 11339
```

## Requesting translation
Install client requirements

```bash
pip3 install -r requirements.txt 
```
Run the client script
```bash
python3 client.py
```

Expected output:
```bash
Text:
He never went out without a book under his arm, and he often came back with two.

Translation:
Il n'est jamais sorti sans un livre sous son bras, et il est souvent revenu avec deux.

```
