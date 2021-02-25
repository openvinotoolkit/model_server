# OpenVINO™ Model Server Stateful Example Clients 

This document contains examples to run *GetModelStatus*, *GetModelMetadata*, *Predict* functions over gRPC API and REST API.

It covers following topics:
* <a href="#grpc-api">gRPC API Stateful Client Example </a>
* <a href="#rest-api">REST API Stateful Client Example </a>

## Requirement

Install client dependencies using the command below in the example_client directory:
```
pip3 install -r stateful_client_requirements.txt
```

### Getting ready with rm_lstm4f stateful model

The purpose of this example is to show how to run stateful models on OpenVino model server.

- Pre-requisite

To run this example you will need to download the rm_lstm4f model with input and score ark files and convert it to IR format.
 1. Download the model from [rm_lstm4f](https://download.01.org/openvinotoolkit/2018_R3/models_contrib/GNA/rm_lstm4f/)

 2. Setup [OpenVINO](https://docs.openvinotoolkit.org/2019_R2/_docs_install_guides_installing_openvino_linux.html)

 3. Convert model to IR [How to convert](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_samples_speech_sample_README.html)

 4. Place ark files in the directories /home/download/rm_lstm4f/test_feat_1_10.ark /home/download/rm_lstm4f/test_score_1_10.ark

Having rm_lstm4f model files .xml and .bin in the IR format present in /home/download/rm_lstm4f/model/1 directory,
OVMS can be started using the command:
```bash
docker run -d -rm -v /home/rm_lstm4f/model:/tmp/model -p 8111:8111 -p 5555:5555 openvino/model_server:latest --stateful --port 8111 --rest_port 5555 --model_name rm_lstm4f --model_path /tmp/model
```

## gRPC API Client Example <a name="grpc-api"></a>

### Predict API 

#### **Submitting gRPC requests based on a dataset from ark files:**

- Command

```bash
usage: grpc_stateful_client.py [--input_path INPUT_PATH]
                              [--score_path SCORE_PATH]
                              [--grpc_address GRPC_ADDRESS]
                              [--grpc_port GRPC_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME]
                              [--debug DEBUG] 
                              [--cw_l CW_L]
                              [--cw_r CW_R]
                              [--sequence_id SEQUENCE_ID]
```

- Arguments

| Argument      | Description |
| :---        |    :----   |
| --input_path   |   Path to input ark file. Default 'rm_lstm4f/test_feat_1_10.ark'|
| --score_path | Path to reference scores ark file. Default 'rm_lstm4f/test_score_1_10.ark' |
| --grpc_address GRPC_ADDRESS | Specify url to grpc service. Default:localhost | 
| --grpc_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --input_name | Specify input tensor name. Default: Parameter |
| --output_name | Specify output name. Default: affinetransform/Fused_Add_ |
| --model_name | Define model name, must be same as is in service. Default: rm_lstm4f|
| --cw_l | Number of requests for left context window. Works only with context window networks. Default: 0 |
| --cw_r | Number of requests for right context window. Works only with context window networks. Default: 0 |
| --debug DEBUG | Enabling debug prints. Set to 1 to enable debug prints. Default: 0 |
| --sequence_id  | Sequence ID used by every sequence provided in ARK files. Setting to 0 means sequence will obtain its ID from OVMS. Default: 0 |


- Usage example

```bash
grpc_stateful_client.py --input_path /home/download/rm_lstm4f/test_feat_1_10.ark --score_path /home/download/rm_lstm4f/test_score_1_10.ark --grpc_address localhost --grpc_port 8111 --input_name Parameter --output_name affinetransform/Fused_Add_ --model_name rm_lstm4f --sequence_id 1

### Starting grpc_stateful_client.py client ###
Context window left width cw_l: 0
Context window right width cw_r: 0
Starting sequence_id: 1
Start processing:
Model name: rm_lstm4f
Reading input ark file /home/download/rm_lstm4f/test_feat_1_10.ark
Reading scores ark file /home/download/rm_lstm4f/test_score_1_10.ark
Adding input name Parameter
Adding output name affinetransform/Fused_Add_
	Sequence name: aem02_st0049_oct89
	Sequence size: 250
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0049_oct89 ; Average RMS Error: 0.0000015199

	Sequence name: aem02_st0122_oct89
	Sequence size: 441
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0122_oct89 ; Average RMS Error: 0.0000015155

	Sequence name: aem02_st0182_oct89
	Sequence size: 347
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0182_oct89 ; Average RMS Error: 0.0000016066

	Sequence name: aem02_st0276_oct89
	Sequence size: 407
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0276_oct89 ; Average RMS Error: 0.0000015985

	Sequence name: aem02_st0343_oct89
	Sequence size: 353
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0343_oct89 ; Average RMS Error: 0.0000017223

	Sequence name: aem02_st0421_oct89
	Sequence size: 372
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0421_oct89 ; Average RMS Error: 0.0000016378

	Sequence name: aem02_st0490_oct89
	Sequence size: 214
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0490_oct89 ; Average RMS Error: 0.0000014651

	Sequence name: aem02_st0554_oct89
	Sequence size: 318
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0554_oct89 ; Average RMS Error: 0.0000017802

	Sequence name: aem02_st0623_oct89
	Sequence size: 285
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0623_oct89 ; Average RMS Error: 0.0000015983

	Sequence name: aem02_st0705_oct89
	Sequence size: 414
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0705_oct89 ; Average RMS Error: 0.0000014956

Global average rms error: 0.0000015940

processing time for all iterations
average time: 2.06 ms; average speed: 201.14 fps
median time: 2.00 ms; median speed: 207.00 fps
max time: 12.00 ms; min speed: 34.50 fps
min time: 1.00 ms; max speed: 414.00 fps
time percentile 90: 2.00 ms; speed percentile 90: 207.00 fps
time percentile 50: 2.00 ms; speed percentile 50: 207.00 fps
time standard deviation: 0.49
time variance: 0.24
### Finished grpc_stateful_client.py client processing ###
```

## REST API Client Example<a name="rest-api"></a>

### Predict API

- Command

```bash
usage: rest_stateful_client.py [--input_path INPUT_PATH]
                              [--score_path SCORE_PATH]
                              [--rest_url REST_URL]
                              [--rest_port REST_PORT]
                              [--input_name INPUT_NAME]
                              [--output_name OUTPUT_NAME]
                              [--model_name MODEL_NAME]
                              [--model_version MODEL_VERSION]
                              [--debug DEBUG] 
                              [--cw_l CW_L]
                              [--cw_r CW_R]
                              [--sequence_id SEQUENCE_ID]
                              [--client_cert CLIENT_CERT]
                              [--client_key CLIENT_KEY]
                              [--ignore_server_verification]
                              [--server_cert SERVER_CERT]
```
- Arguments

| Argument      | Description |
| :---        |    :----   |
| --input_path   |   Path to input ark file. Default 'rm_lstm4f/test_feat_1_10.ark'|
| --score_path | Path to reference scores ark file. Default 'rm_lstm4f/test_score_1_10.ark' |
| --rest_url GRPC_ADDRESS | Specify url to rest service. Default:localhost | 
| --rest_port GRPC_PORT | Specify port to grpc service. Default: 9000 |
| --input_name | Specify input tensor name. Default: Parameter |
| --output_name | Specify output name. Default: affinetransform/Fused_Add_ |
| --model_name | Define model name, must be same as is in service. Default: rm_lstm4f|
| --cw_l | Number of requests for left context window. Works only with context window networks. Default: 0 |
| --cw_r | Number of requests for right context window. Works only with context window networks. Default: 0 |
| --debug DEBUG | Enabling debug prints. Set to 1 to enable debug prints. Default: 0 |
| --sequence_id  | Sequence ID used by every sequence provided in ARK files. Setting to 0 means sequence will obtain its ID from OVMS. Default: 0 |
| --client_cert CLIENT_CERT | Specify mTLS client certificate file. Default: None |
| --client_key CLIENT_KEY | Specify mTLS client key file. Default: None |
| --ignore_server_verification | Skip TLS host verification. Do not use in production. Default: False |
| --server_cert SERVER_CERT | Path to a custom directory containing trusted CA certificates, server certificate, or a CA_BUNDLE file. Default: None, will use default system CA cert store |


- Usage example

```bash
python3 rest_stateful_client.py --input_path /home/download/rm_lstm4f/test_feat_1_10.ark --score_path /home/download/rm_lstm4f/test_score_1_10.ark --rest_url localhost --rest_port 5555 --input_name Parameter --output_name affinetransform/Fused_Add_ --model_name rm_lstm4f --sequence_id 1

### Starting rest_stateful_client.py client ###
Context window left width cw_l: 0
Context window right width cw_r: 0
Starting sequence_id: 1
Start processing:
Model name: rm_lstm4f
Reading input ark file /home/download/rm_lstm4f/test_feat_1_10.ark
Reading scores ark file /home/download/rm_lstm4f/test_score_1_10.ark
Adding input name Parameter
Adding output name affinetransform/Fused_Add_

	Sequence name: aem02_st0049_oct89
	Sequence size: 250
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0049_oct89 ; Average RMS Error: 0.0000015206

	Sequence name: aem02_st0122_oct89
	Sequence size: 441
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0122_oct89 ; Average RMS Error: 0.0000015164

	Sequence name: aem02_st0182_oct89
	Sequence size: 347
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0182_oct89 ; Average RMS Error: 0.0000016075

	Sequence name: aem02_st0276_oct89
	Sequence size: 407
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0276_oct89 ; Average RMS Error: 0.0000015992

	Sequence name: aem02_st0343_oct89
	Sequence size: 353
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0343_oct89 ; Average RMS Error: 0.0000017230

	Sequence name: aem02_st0421_oct89
	Sequence size: 372
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0421_oct89 ; Average RMS Error: 0.0000016386

	Sequence name: aem02_st0490_oct89
	Sequence size: 214
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0490_oct89 ; Average RMS Error: 0.0000014658

	Sequence name: aem02_st0554_oct89
	Sequence size: 318
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0554_oct89 ; Average RMS Error: 0.0000017808

	Sequence name: aem02_st0623_oct89
	Sequence size: 285
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0623_oct89 ; Average RMS Error: 0.0000015989

	Sequence name: aem02_st0705_oct89
	Sequence size: 414
	Sequence id: 1
	Sequence id: 1 ; Sequence name: aem02_st0705_oct89 ; Average RMS Error: 0.0000014963

Global average rms error: 0.0000015947

processing time for all iterations
average time: 4.17 ms; average speed: 99.19 fps
median time: 3.00 ms; median speed: 138.00 fps
max time: 14.00 ms; min speed: 29.57 fps
min time: 2.00 ms; max speed: 207.00 fps
time percentile 90: 7.00 ms; speed percentile 90: 59.14 fps
time percentile 50: 3.00 ms; speed percentile 50: 138.00 fps
time standard deviation: 1.73
time variance: 2.98
### Finished rest_stateful_client.py client processing ###
```
