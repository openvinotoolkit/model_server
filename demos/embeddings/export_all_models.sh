#/bin/bash
# Copyright (c) 2024 Intel Corporation
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
# Execute it in the context of git repository in the demos/embeddings folder
# Add or remove models from the tested_models array to export them

tested_models=(
    nomic-ai/nomic-embed-text-v1.5
    Alibaba-NLP/gte-large-en-v1.5
    BAAI/bge-large-en-v1.5
    BAAI/bge-large-zh-v1.5
    thenlper/gte-small
)
cp config.json config_all.json
cat config_all.json | jq 'del(.mediapipe_config_list[])' | tee config_all.json

for i in "${tested_models[@]}"; do
    echo "$i"
    convert_tokenizer --not-add-special-tokens -o models/$i/tokenizer/1 $i
    optimum-cli export openvino --disable-convert-tokenizer --model $i --task feature-extraction --weight-format int8 --trust-remote-code --library sentence_transformers  models/$i/embeddings/1
    cp models/graph.pbtxt models/$i
    cp subconfig.json models/$i
    sed -i -e "s/\"tokenizer_model\"/\"${i//[\/]/\\/}-tokenizer_model\"/g" models/$i/subconfig.json
    sed -i -e "s/\"embeddings_model\"/\"${i//[\/]/\\/}-embeddings_model\"/g" models/$i/subconfig.json
    sed -i -e "s/servable_name: \"tokenizer_model\"/servable_name: \"${i//[\/]/\\/}-tokenizer_model\"/g" models/$i/graph.pbtxt
    sed -i -e "s/servable_name: \"embeddings_model\"/servable_name: \"${i//[\/]/\\/}-embeddings_model\"/g" models/$i/graph.pbtxt
    cat config_all.json | jq ".mediapipe_config_list[.mediapipe_config_list | length] |= . + {\"name\": \"$i\", \"base_path\": \"models/$i\"}" | tee config_all.json
    python add_rt_info.py --model_path models/$i/tokenizer/1/openvino_tokenizer.xml --config_path models/$i/embeddings/1/tokenizer_config.json
    python add_rt_info.py --model_path models/$i/embeddings/1/openvino_model.xml --config_path models/$i/embeddings/1/config.json
done
