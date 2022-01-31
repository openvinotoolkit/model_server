# BERT Question Answering Demo with OVMS

## Overview

This document demonstrates how tu run inference requests for [BERT model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002) with OpenVINO Model Server.

In this example docker container with [bert-client image](./Dockerfile) runs the script [bert_question_answering.py](./bert_question_answering.py). It runs inference request for each paragraph on a given page in order to answer the provided question. Since each paragraph can have different size the functionality of dynamic shape is used.

NOTE: With `min_request_token_num` parameter you can specify the minimum size of request. If the paragraph has less than required amount it is concatenated with the next one until that requirement is fulfilled. When there is no paragraphs left to concatenate request is created with less amount of tokens than required.

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
[ INFO ] Sequence of length 395 is processed with 2.66 requests/sec (0.38 sec per request)
[ INFO ] Sequence of length 368 is processed with 2.96 requests/sec (0.34 sec per request)
[ INFO ] Sequence of length 32 is processed with 3.65 requests/sec (0.27 sec per request)
[ INFO ] The performance below is reported only for reference purposes, please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.
[ INFO ] 3 requests were processed in 1.00sec (0.33sec per request)
[ INFO ] ---answer: 0.28 Bidirectional Encoder Representations from Transformers
[ INFO ]    Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.[1][2] In 2019, Google announced that it had begun leveraging BERT in its search engine, and by late 2020 it was using BERT in almost every English-language query.  A 2020 literature survey concluded that "in a little over a year, BERT has become a ubiquitous baseline in NLP experiments", counting over 150 research publications analyzing and improving the model.[3]
The original English-language BERT has two models:[1] (1) the BERTBASE: 12 encoders with 12 bidirectional self-attention heads, and (2) the BERTLARGE: 24 encoders with 16 bidirectional self-attention heads. Both models are pre-trained from unlabeled data extracted from the BooksCorpus[4] with 800M words and English Wikipedia with 2,500M words.[5]
BERT is at its core a transformer language model with a variable number of encoder layers and self-attention heads. The architecture is "almost identical" to the original transformer implementation in Vaswani et al. (2017).[6]
BERT was pretrained on two tasks: language modelling (15% of tokens were masked and BERT was trained to predict them from context) and next sentence prediction (BERT was trained to predict if a chosen next sentence was probable or not given the first sentence). As a result of the training process, BERT learns contextual embeddings for words. After pretraining, which is computationally expensive, BERT can be finetuned with less resources on smaller datasets to optimize its performance on specific tasks.[1][7]

[ INFO ] ---answer: 0.26 BERT won the Best Long Paper Award
[ INFO ]    BERT won the Best Long Paper Award at the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL).[20]

[ INFO ] ---answer: 0.19 deeply bidirectional, unsupervised language representation
[ INFO ]    When BERT was published, it achieved state-of-the-art performance on a number of natural language understanding tasks:[1]
The reasons for BERT's state-of-the-art performance on these natural language understanding tasks are not yet well understood.[8][9] Current research has focused on investigating the relationship behind BERT's output as a result of carefully chosen input sequences,[10][11] analysis of internal vector representations through probing classifiers,[12][13] and the relationships represented by attention weights.[8][9]
BERT has its origins from pre-training contextual representations including semi-supervised sequence learning,[14] generative pre-training, ELMo,[15] and ULMFit.[16] Unlike previous models, BERT is a deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary, where BERT takes into account the context for each occurrence of a given word. For instance, whereas the vector for "running" will have the same word2vec vector representation for both of its occurrences in the sentences "He is running a company" and "He is running a marathon", BERT will provide a contextualized embedding that will be different according to the sentence.
On October 25, 2019, Google Search announced that they had started applying BERT models for English language search queries within the US.[17] On December 9, 2019, it was reported that BERT had been adopted by Google Search for over 70 languages.[18] In October 2020, almost every single English-based query was processed by BERT.[19]
```