# Demo for running stable diffusion with OpenVINO Model Server {#ovms_demo_stable_diffusion}


Download the models from HuggingFaces and convert them the ONNX format:

```bash
git clone https://github.com/openvinotoolkit/model_server 
cd model_server/demos/stable-diffusion/python
pip install -r requirements.txt
python vae.py
python text_encoder.py 
python unet.py

``` 

Start OVMS with the models:
```bash
docker run -p 9000:9000 -d -v $(pwd)/:/models openvino/model_server:latest --config_path /models/config.json --port 9000
```

The demo can be execute in 2 modes:
- with local execution using OpenVINO Runtime
- with remote execution via calls to the model server

The executor is abstracted using an adapter which be be selected via `--adapter` parameter.

Local execution:

```bash
python generate.py --adapter openvino --seed 10
Initializing models
Models initialized
Pipeline settings
Input text: sea shore at midnight, epic vista, beach
Negative text: frames, borderline, text, character, duplicate, error, out of frame, watermark
Seed: 10
Number of steps: 20
getting initial random noise (1, 4, 96, 96) {}
100%|████████████████████████████████████████████████| 20/20 [01:15<00:00,  3.75s/it]
```

Remote execution:
```bash
python generate.py --adapter ovms --url localhost:9000 --seed 10
Initializing models
Models initialized
Pipeline settings
Input text: sea shore at midnight, epic vista, beach
Negative text: frames, borderline, text, character, duplicate, error, out of frame, watermark
Seed: 10
Number of steps: 20
getting initial random noise (1, 4, 96, 96) {}
100%|████████████████████████████████████████████████| 20/20 [00:53<00:00,  2.66s/it]

```

In both cases the result is the same and stored in a file `result.png`.
![result](./result.png)

The advantages of using the model server are:
- the client can delegate majority of the pipeline load to a remote host
- the loaded models can be shared with multiple clients

