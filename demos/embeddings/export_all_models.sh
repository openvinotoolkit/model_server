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

mkdir models

for i in "${tested_models[@]}"; do
    echo "$i"
    python export_model.py embeddings --source_model $i --weight-format int8  --config_file_path models/config_all.json
done
