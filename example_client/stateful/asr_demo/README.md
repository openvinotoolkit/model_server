# Speech recognition with ASpIRE Chain Time Delay Neural Network

In this demo you will use OpenVINO Model Server to serve [ASpIRE Chain TDNN](https://kaldi-asr.org/models/m1) model and do speech recognition starting with audio wave file and eding up with text file containing recognized speech.

### 1. Prepare working directories

Create a asr_demo directory in your home catalog with the following 3 subdirectories:
- workspace (used to hold intermediate files)
- data (used to exchange audio, text and model input and output)
- models (used to hold Aspire files for the model server)

```
export WORKSPACE_DIR=$HOME/asr_demo/workspace
export DATA_DIR=$HOME/asr_demo/data
export MODELS_DIR=$HOME/asr_demo/models
mkdir -p $WORKSPACE_DIR $DATA_DIR $MODELS_DIR
```

### 2. Prepare docker images

To successfully run this demo, you will need 3 docker images:
- OpenVINO Model Server image (2021.3 or newer)
- OpenVINO development image
- [Kaldi](https://kaldi-asr.org/) image

Run following commands to obtain them:
```
# Pull OpenVINO images

docker pull openvino/model_server
docker pull openvino/ubuntu18_dev

# Build Kaldi image
cd $WORKSPACE_DIR
wget https://raw.githubusercontent.com/kaldi-asr/kaldi/master/docker/debian10-cpu/Dockerfile
docker build -t kaldi:latest .
```

OpenVINO development image is required to convert Kaldi model to IR format and Kaldi image will help with data processing.

### 3. Prepare ASpIRE TDNN model

Download and unpack the model to `WORKSPACE_DIR`:
```
cd $WORKSPACE_DIR
wget https://kaldi-asr.org/models/1/0001_aspire_chain_model.tar.gz
mkdir aspire_kaldi
tar -xvf 0001_aspire_chain_model.tar.gz -C $WORKSPACE_DIR/aspire_kaldi
```

Use OpenVINO development container with model optimizer to convert model to IR format:

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

As OVMS is already running, you can now convert the wave file to text.
First place the audio speech sample in wave format in `DATA_DIR`. You can use `sample.wav` from this repository:

```
wget https://github.com/openvinotoolkit/model_server/raw/stateful_client_extension/example_client/stateful/asr_demo/sample.wav -O $DATA_DIR/sample.wav
```

and then run:

```
docker run --rm -it --network="host" -v $DATA_DIR:/opt/data -v $MODELS_DIR:/opt/models kaldi:latest bash
```

It will start kaldi container in interactive mode. In the container shell clone model server repository:

```
git clone -b stateful_client_extension https://github.com/openvinotoolkit/model_server.git /opt/model_server
```

Prepare environment for data processing and communication with OVMS:
```
/opt/model_server/example_client/stateful/asr_demo/prepare_environment.sh
```

Run speech recognition:
```
/opt/model_server/example_client/stateful/asr_demo/run.sh
```

Recognized speech should be now present in `DATA_DIR` on your host machine. You can also view it from the container shell as it resides in `/opt/data`:
```
cat /opt/data/sample.wav.txt
sample.wav today we have a very nice weather
```
