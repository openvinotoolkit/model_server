#
# Copyright (c) 2025 Intel Corporation
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

import json

models=[
	"Hermes-3-Llama-3.1-8B",
	"Llama-3.1-8B-Instruct",
	"Mistral-7B-Instruct-v0.3",
	"Phi-4-mini-instruct",
	"Llama-3.2-3B-Instruct",
	"Qwen3-8B",
	"Qwen3-4B",
	"Qwen3-1.7B",
	"Qwen3-0.6B"
]

precisions=["int4","int8", "fp16"]

categories=["simple", "multiple"]

guided_generation = ["true", "false"]

from pathlib import Path
import argparse

base_folder = Path("gorilla/berkeley-function-call-leaderboard")

results = {}


def get_result(base_folder, model, precision, category, guided):
    score = 0.0
    score_file = base_folder / f"{model}-{precision}{guided}_score/ovms-model/BFCL_v3_{category}_score.json"
    print("score file",score_file)
    if score_file.exists():
        with open(score_file, "r") as f:
            first_line = f.readline().strip()
            score_json = json.loads(first_line)
            if "accuracy" in score_json:
                score = score_json["accuracy"]
            print(score)
    return score

parser = argparse.ArgumentParser(description='Process accuracy results')
parser.add_argument('--precisions', nargs='+', default=["int4","int8", "fp16"], help='List of precisions')
parser.add_argument('--categories', nargs='+', default=["simple", "multiple"], help='List of categories')
parser.add_argument('--guided_generation', nargs='+', default=["true", "false"], help='List of guided generation options')
parser.add_argument('--models', nargs='+', default=models, help='List of models')
parser.add_argument('--base_folder', type=str, default="gorilla/berkeley-function-call-leaderboard", help='Base folder path')

args = parser.parse_args()

precisions = args.precisions
categories = args.categories
guided_generation = args.guided_generation
models = args.models
base_folder = Path(args.base_folder)


for each_model in models:
    if each_model not in results:
        results[each_model] = {}
    for each_precision in precisions:
        if each_precision not in results[each_model]:
            results[each_model][each_precision] = {}
        for each_category in categories:
            for guided in guided_generation:
                if each_category not in results[each_model][each_precision]:
                    results[each_model][each_precision][each_category] = {}
                print("Getting result for", each_model, each_precision, each_category, guided)
                results[each_model][each_precision][each_category][guided] = get_result(base_folder, each_model, each_precision, each_category, guided)
                print("res", results)


print(json.dumps(results, indent=2))

print("Enabled tool guided generation")
for each_model in results:
    print(f"{each_model},{results[each_model]['int4']['simple']['true']},{results[each_model]['int4']['multiple']['true']},{results[each_model]['int8']['simple']['true']},{results[each_model]['int8']['multiple']['true']},{results[each_model]['fp16']['simple']['true']},{results[each_model]['fp16']['multiple']['true']}")

print("\nEnabled tool guided generation")
for each_model in results:
    print(f"{each_model},{results[each_model]['int4']['simple']['false']},{results[each_model]['int4']['multiple']['false']},{results[each_model]['int8']['simple']['false']},{results[each_model]['int8']['multiple']['false']},{results[each_model]['fp16']['simple']['false']},{results[each_model]['fp16']['multiple']['false']}")