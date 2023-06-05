# Speech Recognition on Kaldi Model {#ovms_demo_speech_recognition}

This document contains examples to run *Predict* functions over gRPC API and REST API on stateful Kaldi models (rm_lstm4f , aspire_tdnn).

It covers following topics:
* <a href="#grpc-api">gRPC API Stateful Client Example </a>
* <a href="#rest-api">REST API Stateful Client Example </a>

## Requirement

Clone the repository and enter speech_recognition_with_kaldi_model directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/speech_recognition_with_kaldi_model/python
```

Install client dependencies:
```bash
pip3 install -r requirements.txt
```

### Getting ready with rm_lstm4f stateful model

To run this example you will need to download the rm_lstm4f model with input and score ark files and convert it to IR format.
- Download the model from [rm_lstm4f](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/)


Those commands will download necessary files:

```bash
mkdir models && cd models
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/rm_lstm4f.counts
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/rm_lstm4f.nnet
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/rm_lstm4f.mapping
```

rm_lstm4f model files in Kaldi format:

```bash
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/test_feat_1_10.ark
```

[Kaldi's](http://kaldi-asr.org/doc/io.html) binary archive file with input data for the model

```bash
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_lstm4f/test_score_1_10.ark
```

[Kaldi's](http://kaldi-asr.org/doc/io.html) binary archive file with reference model results

- [Convert model to IR](https://docs.openvino.ai/2022.2/openvino_inference_engine_samples_speech_sample_README.html)
 
```bash
docker run -u $(id -u):$(id -g) -v $(pwd):/models:rw openvino/ubuntu20_dev:latest mo --framework kaldi --input_model /models/rm_lstm4f.nnet --counts /models/rm_lstm4f.counts --remove_output_softmax --output_dir /models/rm_lstm4f/1
```

- Having `rm_lstm4f` model files `.xml` and `.bin` in the IR format present in ```bash $(pwd)/rm_lstm4f/1``` directory,
OVMS can be started using the command:

```bash
docker run -d --rm -v $(pwd)/rm_lstm4f/:/tmp/model -p 9000:9000 -p 5555:5555 openvino/model_server:latest --stateful --port 9000 --rest_port 5555 --model_name rm_lstm4f --model_path /tmp/model
```

- Return to the demo root directory

```bash
cd ..
```

## gRPC API Client Example <a name="grpc-api"></a>

### Predict API 

#### **Submitting gRPC requests based on a dataset from ark files:**

- Command

```bash
python grpc_stateful_client.py --help
usage: grpc_stateful_client.py [-h] [--input_path INPUT_PATH]
                              [--score_path SCORE_PATH]
                              [--output_path OUTPUT_PATH]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME] [--debug DEBUG] 
                              [--cw_l CW_L] [--cw_r CW_R]
                              [--sequence_id SEQUENCE_ID]
```

- Arguments

| Argument      | Description |
| :---        |    :----   |
| --input_path   |   Path to input ark file. Default: ```rm_lstm4f/test_feat_1_10.ark```|
| --score_path | Path to reference scores ark file. Default: ```rm_lstm4f/test_score_1_10.ark``` |
| --grpc_address GRPC_ADDRESS | Specify url to grpc service. Default:```localhost``` | 
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: ```9000``` |
| --input_name | Specify input tensor name. Default: ```Parameter:0``` |
| --output_name | Specify output name. Default: ```affinetransform:0``` |
| --model_name | Define model name, must be same as is in service. Default: ```rm_lstm4f```|
| --cw_l | Number of requests for left context window. Works only with context window networks. Default: ```0``` |
| --cw_r | Number of requests for right context window. Works only with context window networks. Default: ```0``` |
| --debug DEBUG | Enabling debug prints. Set to 1 to enable debug prints. Default: ```0``` |
| --sequence_id  | Sequence ID used by every sequence provided in ARK files. Setting to 0 means sequence will obtain its ID from OVMS. Default: ```0``` |


- Usage example

```bash
python3 grpc_stateful_client.py --input_path models/test_feat_1_10.ark --score_path models/test_score_1_10.ark --grpc_address localhost --grpc_port 9000 --input_name Parameter:0 --output_name affinetransform:0 --model_name rm_lstm4f --sequence_id 1

### Starting grpc_stateful_client.py client ###
Context window left width cw_l: 0
Context window right width cw_r: 0
Starting sequence_id: 1
Start processing:
Model name: rm_lstm4f
Reading input ark file models/test_feat_1_10.ark
Reading scores ark file models/test_score_1_10.ark
Adding input name Parameter:0
Adding output name affinetransform:0

        Sequence name: aem02_st0049_oct89
        Sequence size: 250
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0049_oct89 ; Average RMS Error: 0.0000022994


        Sequence name: aem02_st0122_oct89
        Sequence size: 441
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0122_oct89 ; Average RMS Error: 0.0000021833


        Sequence name: aem02_st0182_oct89
        Sequence size: 347
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0182_oct89 ; Average RMS Error: 0.0000024054


        Sequence name: aem02_st0276_oct89
        Sequence size: 407
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0276_oct89 ; Average RMS Error: 0.0000024259


        Sequence name: aem02_st0343_oct89
        Sequence size: 353
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0343_oct89 ; Average RMS Error: 0.0000025725


        Sequence name: aem02_st0421_oct89
        Sequence size: 372
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0421_oct89 ; Average RMS Error: 0.0000021470


        Sequence name: aem02_st0490_oct89
        Sequence size: 214
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0490_oct89 ; Average RMS Error: 0.0000020102


        Sequence name: aem02_st0554_oct89
        Sequence size: 318
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0554_oct89 ; Average RMS Error: 0.0000023627


        Sequence name: aem02_st0623_oct89
        Sequence size: 285
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0623_oct89 ; Average RMS Error: 0.0000023717


        Sequence name: aem02_st0705_oct89
        Sequence size: 414
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0705_oct89 ; Average RMS Error: 0.0000021335

Global average rms error: 0.0000022912


processing time for all iterations
average time: 1.10 ms; average speed: 375.77 fps
median time: 1.00 ms; median speed: 414.00 fps
max time: 18.00 ms; min speed: 23.00 fps
min time: 0.00 ms; max speed: inf fps
time percentile 90: 1.00 ms; speed percentile 90: 414.00 fps
time percentile 50: 1.00 ms; speed percentile 50: 414.00 fps
time standard deviation: 0.52
time variance: 0.27
### Finished grpc_stateful_client.py client processing ###
```

## REST API Client Example<a name="rest-api"></a>

### Predict API

- Command

```bash
python rest_stateful_client.py --help
usage: rest_stateful_client.py [-h] [--input_path INPUT_PATH]
                              [--score_path SCORE_PATH]
                              [--rest_port REST_PORT] [--rest_url REST_URL]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME] [--debug DEBUG]
                              [--cw_l CW_L] [--cw_r CW_R]
                              [--sequence_id SEQUENCE_ID]
                              [--model_version MODEL_VERSION]
                              [--client_cert CLIENT_CERT]
                              [--client_key CLIENT_KEY]
                              [--ignore_server_verification]
                              [--server_cert SERVER_CERT]
```
- Arguments

| Argument      | Description |
| :---        |    :----   |
| --input_path   |   Path to input ark file. Default: ```rm_lstm4f/test_feat_1_10.ark```|
| --score_path | Path to reference scores ark file. Default: ```rm_lstm4f/test_score_1_10.ark``` |
| --rest_url REST_URL | Specify url to rest service. Default: ```localhost``` | 
| --rest_port REST_PORT | Specify port to grpc service. Default: ```9000``` |
| --input_name | Specify input tensor name. Default: ```Parameter:0``` |
| --output_name | Specify output name. Default: ```affinetransform:0``` |
| --model_name | Define model name, must be same as is in service. Default: ```rm_lstm4f```|
| --cw_l | Number of requests for left context window. Works only with context window networks. Default: ```0``` |
| --cw_r | Number of requests for right context window. Works only with context window networks. Default: ```0``` |
| --debug DEBUG | Enabling debug prints. Set to 1 to enable debug prints. Default: ```0``` |
| --sequence_id  | Sequence ID used by every sequence provided in ARK files. Setting to 0 means sequence will obtain its ID from OVMS. Default: ```0``` |
| --client_cert CLIENT_CERT | Specify mTLS client certificate file. Default: ```None``` |
| --client_key CLIENT_KEY | Specify mTLS client key file. Default: ```None``` |
| --ignore_server_verification | Skip TLS host verification. Do not use in production. Default: ```False``` |
| --server_cert SERVER_CERT | Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. Default: ```None```, will use default system CA cert store |


- Usage example

```bash
python3 rest_stateful_client.py --input_path models/test_feat_1_10.ark --score_path models/test_score_1_10.ark --rest_url localhost --rest_port 5555 --input_name Parameter:0 --output_name affinetransform:0 --model_name rm_lstm4f --sequence_id 1

### Starting rest_stateful_client.py client ###
Context window left width cw_l: 0
Context window right width cw_r: 0
Starting sequence_id: 1
Start processing:
Model name: rm_lstm4f
Reading input ark file models/test_feat_1_10.ark
Reading scores ark file models/test_score_1_10.ark
Adding input name Parameter:0
Adding output name affinetransform:0

        Sequence name: aem02_st0049_oct89
        Sequence size: 250
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0049_oct89 ; Average RMS Error: 0.0000022999


        Sequence name: aem02_st0122_oct89
        Sequence size: 441
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0122_oct89 ; Average RMS Error: 0.0000021839


        Sequence name: aem02_st0182_oct89
        Sequence size: 347
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0182_oct89 ; Average RMS Error: 0.0000024058


        Sequence name: aem02_st0276_oct89
        Sequence size: 407
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0276_oct89 ; Average RMS Error: 0.0000024263


        Sequence name: aem02_st0343_oct89
        Sequence size: 353
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0343_oct89 ; Average RMS Error: 0.0000025731


        Sequence name: aem02_st0421_oct89
        Sequence size: 372
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0421_oct89 ; Average RMS Error: 0.0000021476


        Sequence name: aem02_st0490_oct89
        Sequence size: 214
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0490_oct89 ; Average RMS Error: 0.0000020109


        Sequence name: aem02_st0554_oct89
        Sequence size: 318
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0554_oct89 ; Average RMS Error: 0.0000023632


        Sequence name: aem02_st0623_oct89
        Sequence size: 285
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0623_oct89 ; Average RMS Error: 0.0000023723


        Sequence name: aem02_st0705_oct89
        Sequence size: 414
        Sequence id: 1
        Sequence id: 1 ; Sequence name: aem02_st0705_oct89 ; Average RMS Error: 0.0000021341

Global average rms error: 0.0000022917


processing time for all iterations
average time: 5.04 ms; average speed: 82.21 fps
median time: 5.00 ms; median speed: 82.80 fps
max time: 17.00 ms; min speed: 24.35 fps
min time: 4.00 ms; max speed: 103.50 fps
time percentile 90: 5.00 ms; speed percentile 90: 82.80 fps
time percentile 50: 5.00 ms; speed percentile 50: 82.80 fps
time standard deviation: 0.40
time variance: 0.16
### Finished rest_stateful_client.py client processing ###
```
