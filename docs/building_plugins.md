# To build CUDA and Auto plugins execute following commands inside plugins directory:

1. 
```bash
   git clone https://github.com/openvinotoolkit/openvino_cuda_plugin.git /openvino_contrib/
   cd openvino_contrib
   git submodule update --init --recursive
   cd ..
```

2. 
```bash
   git clone -b huyuan/2021_4 https://github.com.intel-sandbox/iotg-odt-prc-openvino.git
   cd iotg-odt-prc-openvino
   git submodule update --init --recursive
   cd ..
```

3. Download cuDNN and cuTENSOR packages and place them inside plugins dir 
   (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html, https://docs.nvidia.com/cuda/cutensor/getting_started.html#installation-and-compilation ) 

4. 
```bash
   git clone -b huyuan/2021_4 https://github.com.intel-sandbox/iotg-odt-prc-openvino.git
   cd iotg-odt-prc-openvino
   git submodule update --init --recursive
   cd ..
```

5. 
```bash
   docker build -t openvino/plugins . 
```

libCUDAPlugin.so, libAutoPlugin.so, libinterpreter_backend.so and libngraph_backend.so can be found at /plugins inside docker container