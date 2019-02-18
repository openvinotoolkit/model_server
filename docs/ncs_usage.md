# Using Neural Compute Sticks

OpenVINO Model Server supports AI accelerators Intel® Neural Compute Stick and Intel® Neural 
Compute Stick 2. __Learn more about Neural Compute Sticks [here](https://software.intel.com/en-us/neural-compute-stick).__

NCS can be used with OVMS as soon as it works with OpenVINO. You need to have OpenVINO Toolkit 
with Movidius VPU support installed, some environment variables set and NCS USB Driver configured.
In order to do that follow instruction provided [here](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).

After successfully completing these steps you can try running OpenVINO Model Server with NCS or 
NCS2. Before starting server, you need to specify that you want to load model on Neural Compute 
Stick. You can do that by setting environment variable <i>DEVICE</i> to <i>MYRIAD</i>. If it's not 
specified, OpenVINO will try to load model on CPU.

Example:
```
export DEVICE=MYRIAD

ie_serving model --model_path /opt/model --model_name my_model --port 9001
```

You can also [run it in Docker container](docker_container.md)

**Note**: Currently Neural Computing Sticks support only **FP16** target precision. Make sure you 
have proper model. If not, take a look at [OpenVINO Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) 
and convert your model to desired format.
