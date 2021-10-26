# BERT model usage example with OVMS

## Starting OVMS with BERT model

```bash
curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/bert-large-uncased-whole-word-masking-squad-int8-0001/FP32-INT8/bert-large-uncased-whole-word-masking-squad-int8-0001.bin https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/2/bert-large-uncased-whole-word-masking-squad-int8-0001/FP32-INT8/bert-large-uncased-whole-word-masking-squad-int8-0001.xml -o model/1/bert-large-uncased-whole-word-masking-squad-int8-0001.bin -o model/1/bert-large-uncased-whole-word-masking-squad-int8-0001.xml
chmod -R 755 model
docker run -d -v $(pwd)/model:/models -p 4000:4000 openvino/model_server:latest  --model_path /models --model_name bert --port 4000  --shape auto
```

## Starting BERT client
```
docker build -t bert-client:latest .

export SERVER_IP=<external IP>
docker run -it -e no_proxy=${SERVER_IP} bert-client:latest --grpc_address ${SERVER_IP}
```

Docker image with BERT client by default start the container with a command:
```
python bert_question_answering_demo_ovms.py -v vocab.txt -i https://en.wikipedia.org/wiki/Graph_theory --question "what is a graph" --grpc_port 4000 --input_names result.1,result.2,result.3 --output_names 5211,5212
```
You can change the entrypoint to adjust to different parameters

Example of the output snippet:
```bash
[ INFO ] Sequence of length 384 is processed with 3.39 requests/sec (0.3 sec per request)
c_e 5582 c_s 5264
[ INFO ] Sequence of length 384 is processed with 3.51 requests/sec (0.28 sec per request)
[ INFO ] The performance below is reported only for reference purposes, please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.
[ INFO ] 29 requests of 384 length were processed in 9.17sec (0.32sec per request)
[ INFO ] ---answer: 0.52 an ordered triple
[ INFO ]    In one more general sense of the term allowing multiple edges,[3][4] a graph is an ordered triple
[ INFO ] ---answer: 0.39 analysis of language as a graph
[ INFO ]     finite-state morphology, using finite-state transducers) are common in the analysis of language as a graph
[ INFO ] ---answer: 0.34 For a planar graph, the crossing number is zero by definition
[ INFO ]     For a planar graph, the crossing number is zero by definition
question: what is a graph
```