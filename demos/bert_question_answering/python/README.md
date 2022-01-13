# BERT Question Answering Demo with OVMS

## Starting OVMS with BERT model

```bash
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/bert-small-uncased-whole-word-masking-squad-int8-0002/FP32-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/bert-small-uncased-whole-word-masking-squad-int8-0002/FP32-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.xml -o model/1/bert-small-uncased-whole-word-masking-squad-int8-0002.bin -o model/1/bert-small-uncased-whole-word-masking-squad-int8-0002.xml
chmod -R 755 model
docker run -d -v $(pwd)/model:/models -p 9000:9000 openvino/model_server:latest  --model_path /models --model_name bert --port 9000  --shape auto
```

## Starting BERT client
```
docker build -t bert-client:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} .

docker run -it --network host -e no_proxy=localhost bert-client:latest --grpc_address localhost
```

Docker image with BERT client by default start the container with a command:
```
python bert_question_answering_demo_ovms.py -v vocab.txt -i "https://en.wikipedia.org/wiki/BERT_(language_model)" --question "what is bert" --grpc_port 4000 --input_names input_ids,attention_mask,token_type_ids,position_ids --output_names output_s,output_e
```
You can change the entrypoint to adjust to different parameters

Example of the output snippet:
```bash
question: what is bert
c_e 378 c_s 0
[ INFO ] Sequence of length 384 is processed with 14.75 requests/sec (0.068 sec per request)
c_e 567 c_s 189
[ INFO ] Sequence of length 384 is processed with 15.36 requests/sec (0.065 sec per request)
c_e 756 c_s 378
[ INFO ] Sequence of length 384 is processed with 29.01 requests/sec (0.034 sec per request)
c_e 794 c_s 567
[ INFO ] Sequence of length 384 is processed with 29.28 requests/sec (0.034 sec per request)
[ INFO ] The performance below is reported only for reference purposes, please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.
[ INFO ] 4 requests of 384 length were processed in 0.21sec (0.052sec per request)
[ INFO ] ---answer: 0.29 Bidirectional Encoder Representations from Transformers
[ INFO ]    Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google
[ INFO ] ---answer: 0.15 deeply bidirectional, unsupervised language representation
[ INFO ]    [16] Unlike previous models, BERT is a deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus
[ INFO ] ---answer: 0.11 self-attention heads
[ INFO ]    BERT is at its core a Transformer language model with variable number of encoder layers and self-attention heads
```