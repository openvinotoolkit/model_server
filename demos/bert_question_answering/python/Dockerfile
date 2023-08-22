#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


FROM ubuntu:20.04
RUN apt update && apt install -y python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /bert
COPY bert_question_answering.py tokens_bert.py html_reader.py requirements.txt ./
ADD https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/bert-small-uncased-whole-word-masking-squad-int8-0002/vocab.txt .
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "bert_question_answering.py", "-v", "vocab.txt", "-i", "https://en.wikipedia.org/w/index.php?title=BERT_(language_model)&oldid=1148859098", "--question", "what is bert", "--grpc_port", "9000", "--input_names", "input_ids,attention_mask,token_type_ids,position_ids"]
