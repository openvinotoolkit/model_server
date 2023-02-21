# BERT Question Answering Demo {#ovms_demo_bert}

## Overview

This document demonstrates how to run inference requests for [BERT model](https://github.com/openvinotoolkit/open_model_zoo/tree/2022.1.0/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002) with OpenVINO Model Server. It provides questions answering functionality.

In this example docker container with [bert-client image](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/bert_question_answering/python/Dockerfile) runs the script [bert_question_answering.py](https://github.com/openvinotoolkit/model_server/blob/releases/2022/1/demos/bert_question_answering/python/bert_question_answering.py). It runs inference request for each paragraph on a given page in order to answer the provided question. Since each paragraph can have different size the functionality of dynamic shape is used.

NOTE: With `min_request_token_num` parameter you can specify the minimum size of the request. If the paragraph has too short, it is concatenated with the next one until it has required length. When there is no paragraphs left to concatenate request is created with the remaining content.

## Starting OVMS with BERT model

```bash
curl --create-dirs https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/bert-small-uncased-whole-word-masking-squad-int8-0002/FP32-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/bert-small-uncased-whole-word-masking-squad-int8-0002/FP32-INT8/bert-small-uncased-whole-word-masking-squad-int8-0002.xml -o model/1/bert-small-uncased-whole-word-masking-squad-int8-0002.bin -o model/1/bert-small-uncased-whole-word-masking-squad-int8-0002.xml
chmod -R 755 model
docker run -d -v $(pwd)/model:/models -p 9000:9000 openvino/model_server:latest  --model_path /models --model_name bert --port 9000 --shape '{"attention_mask": "(1,-1)", "input_ids": "(1,-1)", "position_ids": "(1,-1)", "token_type_ids": "(1,-1)"}'
```

## Starting BERT client
Clone the repository and enter bert_question_answering directory
```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/demos/bert_question_answering/python
```
Build and start the docker container which runs the client
```bash
docker build -t bert-client:latest --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy} .

docker run -it --network host -e no_proxy=localhost bert-client:latest --grpc_address localhost --grpc_port 9000
```

Docker image with BERT client by default start the container with a command:
```
python bert_question_answering.py -v vocab.txt -i "https://en.wikipedia.org/wiki/BERT_(language_model)" --question "what is bert" --grpc_port 9000 --input_names input_ids,attention_mask,token_type_ids,position_ids
```
You can change the entrypoint to adjust to different parameters

Example of the output snippet:
```bash
question: what is bert
[ INFO ] Sequence of length 343 is processed with 6.14 requests/sec (0.16 sec per request)
[ INFO ] Sequence of length 370 is processed with 9.21 requests/sec (0.11 sec per request)
[ INFO ] Sequence of length 36 is processed with 16.17 requests/sec (0.062 sec per request)
[ INFO ] The performance below is reported only for reference purposes, please use the benchmark_app tool (part of the OpenVINO samples) for any actual measurements.
[ INFO ] 3 requests were processed in 0.36sec (0.12sec per request)
[ INFO ] ---answer: 0.68 The research paper describing BERT
[ INFO ]    The research paper describing BERT won the Best Long Paper Award at the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL).[21]

[ INFO ] ---answer: 0.23 Bidirectional Encoder Representations from Transformers
[ INFO ]    Bidirectional Encoder Representations from Transformers (BERT) is a family of masked-language models published in 2018 by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.[1][2] A 2020 literature survey concluded that "in a little over a year, BERT has become a ubiquitous baseline in NLP experiments", counting over 150 research publications analyzing and improving the model.[3]
BERT was originally implemented in the English language at two model sizes:[1] (1) BERTBASE: 12 encoders with 12 bidirectional self-attention heads totaling 110 million parameters, and (2) BERTLARGE: 24 encoders with 16 bidirectional self-attention heads totaling 340 million parameters. Both models were pre-trained on the Toronto BooksCorpus[4] (800M words) and English Wikipedia  (2,500M words).
BERT is based on the transformer architecture. Specifically, BERT is composed of Transformer encoder layers.
BERT was pre-trained simultaneously on two tasks:  language modeling (15% of tokens were masked, and the training objective was to predict the original token given its context) and next sentence prediction (the training objective was to classify if two spans of text appeared sequentially in the training corpus).[5] As a result of this training process, BERT learns latent representations of words and sentences in context. After pre-training, BERT can be finetuned with fewer resources on smaller datasets to optimize its performance on specific tasks.[1][6] The pre-training stage is significantly more computationally expensive than finetuning.

[ INFO ] ---answer: 0.16 a deeply bidirectional, unsupervised language representation
[ INFO ]    When BERT was published, it achieved state-of-the-art performance on a number of natural language understanding tasks:[1]
The reasons for BERT's state-of-the-art performance on these natural language understanding tasks are not yet well understood.[9][10] Current research has focused on investigating the relationship behind BERT's output as a result of carefully chosen input sequences,[11][12] analysis of internal vector representations through probing classifiers,[13][14] and the relationships represented by attention weights.[9][10]
BERT has its origins from pre-training contextual representations, including semi-supervised sequence learning,[15] generative pre-training, ELMo,[16] and ULMFit.[17] Unlike previous models, BERT is a deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary, where BERT takes into account the context for each occurrence of a given word. For instance, whereas the vector for "running" will have the same word2vec vector representation for both of its occurrences in the sentences "He is running a company" and "He is running a marathon", BERT will provide a contextualized embedding that will be different according to the sentence.
On October 25, 2019, Google announced that they had started applying BERT models for English language search queries within the US.[18] On December 9, 2019, it was reported that BERT had been adopted by Google Search for over 70 languages.[19] In October 2020, almost every single English-based query was processed by a BERT model.[20]
```
