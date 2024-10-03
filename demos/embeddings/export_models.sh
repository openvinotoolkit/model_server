#!/bin/bash
## prepare env ##
python3.10 -m venv .venv_dk
. .venv_dk/bin/activate
pip3 install -U pip
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly"
pip3 install --pre "optimum-intel[nncf,openvino]"@git+https://github.com/huggingface/optimum-intel.git openvino-tokenizers
## ### ### ### ##

# vvvv embeddings
optimum-cli export openvino --model BAAI/bge-large-en-v1.5 --task feature-extraction bge-large-en-v1.5_embeddings
convert_tokenizer -o bge-large-en-v1.5_tokenizer --skip-special-tokens BAAI/bge-large-en-v1.5

# vvvv re-ranker, contains tokenizer and detokenizer inside dir
optimum-cli export openvino --model BAAI/bge-reranker-large  bge-reranker-large --trust-remote-code

# needed to install sentence_transformers, worked
# vvvvvvvvvvvvv embeddings
pip install sentence_transformers
optimum-cli export openvino -m Alibaba-NLP/gte-large-en-v1.5 --library sentence_transformers --task feature-extraction gte-large-en-v1.5 --trust-remote-code

# vvvvvvvvvv embeddings
# must install flash_attn, cannot install
#pip install flash_attn # ??????????????????????????/
# Requires CUDA to download.
# DO NOT USE PIP_EXTRA_INDEX_URL to install torch
# Requires CUDA drivers 11.6++
# And it requires NVIDIA GPU on host?
# update pip, wheel
#pip install flash_attn # ??????????????????????????/
optimum-cli export openvino -m Alibaba-NLP/gte-Qwen2-7B-instruct --library sentence_transformers --task feature-extraction gte-Qwen2-7B-instruct --trust-remote-code
