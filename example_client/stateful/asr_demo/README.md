# Speech recognition with ASpIRE Chain Time Delay Neural Network

In this demo you will use OpenVINO Model Server to serve [ASpIRE Chain TDNN](https://kaldi-asr.org/models/m1) model and do speech recognition starting with audio wave file and eding up with text file containing recognized speech. Presented steps serve as a demonstration of inference on stateful model and should not be considered as production setup.

### 1. Prepare working directories

Create a asr_demo directory in your home catalog with the following 3 subdirectories:
- workspace (used to hold intermediate files)
- models (used to hold Aspire files for the model server)

```
export WORKSPACE_DIR=$HOME/asr_demo/workspace
export MODELS_DIR=$HOME/asr_demo/models
mkdir -p $WORKSPACE_DIR $MODELS_DIR
```

### 2. Prepare docker images

To successfully run this demo, you will need 3 docker images:
- OpenVINO Model Server image (2021.3 or newer)
- OpenVINO development image
- [Kaldi](https://kaldi-asr.org/) image

Pull OpenVINO and OpenVINO Model Server images:
```
docker pull openvino/model_server
docker pull openvino/ubuntu18_dev
```

Download Kaldi dockerfile and modify it to contain necessary dependencies:

```
cd $WORKSPACE_DIR

wget https://raw.githubusercontent.com/kaldi-asr/kaldi/master/docker/debian10-cpu/Dockerfile

sed -i '$d' Dockerfile

echo '
RUN cd /opt/kaldi/egs/aspire/s5 && \
    wget https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2 && \
    tar -xvf 0001_aspire_chain_model_with_hclg.tar.bz2 && \
    rm -f 0001_aspire_chain_model_with_hclg.tar.bz2

RUN apt-get install -y virtualenv

RUN git clone -b stateful_client_extension https://github.com/openvinotoolkit/model_server.git /opt/model_server && \
    cd /opt/model_server && \
    virtualenv -p python3 .venv && \
    . .venv/bin/activate && \
    pip install tensorflow-serving-api==2.* kaldi-python-io==1.2.1 && \
    echo "source /opt/model_server/.venv/bin/activate" | tee -a /root/.bashrc && \
    mkdir /opt/workspace

WORKDIR /opt/workspace/
' >> Dockerfile

docker build -t kaldi:latest .
```

OpenVINO development image is required to convert Kaldi model to IR format.
Model server container will be used as an inference service and Kaldi container will be used as a OVMS client.

### 3. Prepare ASpIRE TDNN model

Download and unpack the model to `WORKSPACE_DIR`:
```
cd $WORKSPACE_DIR
wget https://kaldi-asr.org/models/1/0001_aspire_chain_model.tar.gz
mkdir aspire_kaldi
tar -xvf 0001_aspire_chain_model.tar.gz -C $WORKSPACE_DIR/aspire_kaldi
```

Use temporary OpenVINO development container with model optimizer to convert model to IR format:

```
docker run --rm -it -u 0 -v $WORKSPACE_DIR:/opt/workspace openvino/ubuntu18_dev python3 /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/mo_kaldi.py --input_model /opt/workspace/aspire_kaldi/exp/chain/tdnn_7b/final.mdl --output output --output_dir /opt/workspace
```

After running this command you should have `final.xml` and `final.bin` files in your `WORKSPACE_DIR` directory.

### 4. Prepare models directory to work with OVMS

Now that you have the model, you need to create correct directory structure for model server to read the model. First create `aspire` model directory with version `1` in `MODELS_DIR`:

```
mkdir -p $MODELS_DIR/aspire/1
```

Then copy `final.xml` and `final.bin` from the `WORKSPACE_DIR` to newly created directory:

```
cp $WORKSPACE_DIR/final.xml $WORKSPACE_DIR/final.bin $MODELS_DIR/aspire/1
```

### 5. Start OpenVINO Model Server

When models directory is ready you can start OVMS with a command:

```
docker run --rm -d -p 9000:9000 -v $MODELS_DIR:/opt/models openvino/model_server:latest --model_name aspire --model_path /opt/models/aspire --port 9000 --stateful
```

### 6. Do speech recognition

As OVMS is already running in the background, you need to start another container that will be the client.
Start kaldi container built in the step 2 in interactive mode:

```
docker run --rm -it --network="host" kaldi:latest bash
```

The container constains everything required for data processing and communication with the model server.
It runs with `host` network parameter to make it easy to access model server container running on the same host.
As you start the container, the working directory is `/opt/workspace`

Download the sample audio file for speech recognition:

```
wget https://github.com/openvinotoolkit/model_server/raw/stateful_client_extension/example_client/stateful/asr_demo/sample.wav
```

Run speech recognition:
```
/opt/model_server/example_client/stateful/asr_demo/run.sh /opt/workspace/sample.wav localhost 9000
```

The `run.sh` script takes 3 arguments, first is the absolute path to the `wav` file, second is the OVMS address and the third is the port on which model server listens.
At the beginning the MFCC features and ivectors are extracted, then they are sent to the model server via [grpc_stateful_client](../grpc_stateful_client.py). At the end the results are decoded and parsed into text.

When the command finishes successfully you should see the `txt` file in the same directory as the `wav` one:
```
cat /opt/workspace/sample.wav.txt
/opt/workspace/sample.wav today we have a very nice weather
```
