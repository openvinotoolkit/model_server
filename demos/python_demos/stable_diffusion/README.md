## Download model

We are going to use stable-diffusion model in this scenario.  
Download the model using `download_model.py` script:

```bash
pip install -r requirements.txt
python3 download_model.py
```

The model will appear in `./model` directory.

## Run Model Server

Mount the `./model` directory with the model.  
Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and uses optimum-intel framework
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

```bash
docker run -it --rm -p 11339:11339 -v $(pwd)/servable:/workspace:rw -v $(pwd)/model:/model/:rw -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy <IMAGE> --config_path /workspace/config.json --log_level DEBUG --port 11339
```

Replace `<IMAGE>` with pre-built OVMS image containing required pip packages.

## Sending request to the model

The client script contains hardcoded prompt:
```
Zebras in space
```

Run client script:
```bash
python3 client.py
```

Output image will be saved as output.png
