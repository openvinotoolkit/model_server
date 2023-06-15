# Demo for running stable diffusion with OpenVINO Model Server


Download the models from HuggingFaces and convert them the ONNX format:

```bash
python vae.py
python text_encoder.py 
python unet.py

``` 

Start OVMS server with the models:
```bash
docker run -p 15000:15000 -p 15001:15001 -d -v /home/dtrawins/stable-diffusion/unet/:/model stable-diffusion:latest --model_name unet --model_path /model --log_level DEBUG --layout '{"latent_model_input":"...:NCHW","t":"...:...","encoder_hidden_states":"...:CHW"}' --port 15000 --rest_port 15001
```

Edit in the script stable_diffusion.py the input parameters: prompt, negative prompt, seed and number_of_steps.
Run the pipeline:

```bash
python stable_diffusion.py

```

![result](./result.png)
