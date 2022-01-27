# BERT Question Answering Demo with OVMS

## Starting OVMS with BERT model

```bash
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/bert-small-uncased-whole-word-masking-squad-int8-0002/FP32-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3/bert-small-uncased-whole-word-masking-squad-int8-0002/FP32-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.xml -o model/1/bert-small-uncased-whole-word-masking-squad-int8-0002.bin -o model/1/bert-small-uncased-whole-word-masking-squad-int8-0002.xml
chmod -R 755 model
docker run -d -v $(pwd)/model:/models -p 9000:9000 openvino/model_server:latest  --model_path /models --model_name bert --port 9000  --shape '{"attention_mask": "(1,-1)", "input_ids": "(1,-1)", "position_ids": "(1,-1)", "token_type_ids": "(1,-1)"}'
```

## Starting BERT client
```
docker build -t bert-client:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} .

docker run -it --network host -e no_proxy=localhost bert-client:latest --grpc_address localhost
```

Docker image with BERT client by default start the container with a command:
```
python bert_question_answering.py -v vocab.txt -i "https://en.wikipedia.org/wiki/BERT_(language_model)" --question "what is bert" --grpc_port 4000 --input_names input_ids,attention_mask,token_type_ids,position_ids
```
You can change the entrypoint to adjust to different parameters

Example of the output snippet:
```bash
question: what is bert
[ INFO ] Sequence of length 132 is processed with 3.83 requests/sec (0.26 sec per request)
[ INFO ] Sequence of length 97 is processed with 6.50 requests/sec (0.15 sec per request)
[ INFO ] Sequence of length 52 is processed with 9.38 requests/sec (0.11 sec per request)
[ INFO ] Sequence of length 114 is processed with 5.95 requests/sec (0.17 sec per request)
[ INFO ] Sequence of length 27 is processed with 13.09 requests/sec (0.076 sec per request)
[ INFO ] Sequence of length 90 is processed with 8.75 requests/sec (0.11 sec per request)
[ INFO ] Sequence of length 178 is processed with 3.97 requests/sec (0.25 sec per request)
[ INFO ] Sequence of length 73 is processed with 10.06 requests/sec (0.099 sec per request)
[ INFO ] Sequence of length 32 is processed with 12.92 requests/sec (0.077 sec per request)
[ INFO ] The performance below is reported only for reference purposes, please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.
[ INFO ] 9 requests were processed in 1.33sec (0.15sec per request)
[ INFO ] ---answer: 0.46 computationally expensive
[ INFO ]     After pretraining, which is computationally expensive, BERT can be finetuned with less resources on smaller datasets to optimize its performance on specific tasks
[ INFO ] ---answer: 0.26 BERT won the Best Long Paper Award
[ INFO ]    BERT won the Best Long Paper Award at the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)
[ INFO ] ---answer: 0.26 The original English-language BERT has two models
[ INFO ]    The original English-language BERT has two models:[1] (1) the BERTBASE: 12 encoders with 12 bidirectional self-attention heads, and (2) the BERTLARGE: 24 encoders with 16 bidirectional self-attention heads
[ INFO ] ---answer: 0.23 Bidirectional Encoder Representations from Transformers
[ INFO ]    Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google
[ INFO ] ---answer: 0.14 BERT is at its core a transformer language model
[ INFO ]    BERT is at its core a transformer language model with a variable number of encoder layers and self-attention heads
```