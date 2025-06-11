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


# This script generates a dataset of long context examples for performance evaluation
import os
import json
import requests
from transformers import AutoTokenizer
import argparse

# function to download a file from a URL and convert it to text
def download_file(url):
    output_path = os.path.basename(url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved as {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    if url.endswith(".txt"):
        with open(output_path, "r", encoding="utf-8") as file:
            text = file.read()
        print(f"Text file read successfully. Length of text: {len(text)} characters")
        return text
    elif url.endswith(".pdf"):
        with open(output_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    else:
        print("Unsupported file type. Only .txt and .pdf files are supported.")

parser = argparse.ArgumentParser(description="Generate a dataset of long context examples.")
parser.add_argument("--file_url", type=str, default="https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2011/donquix-2011.txt", help="URL of the file to download")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M", help="Model name for the tokenizer")
parser.add_argument("--limit_context_tokens", type=int, default=50000, help="Maximum number of tokens to use for the context")
args = parser.parse_args()

file_url = args.file_url
model_name = args.model_name
limit_context_tokens = args.limit_context_tokens    

file_url = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2011/donquix-2011.txt"

text = download_file(file_url)

# Initialize the tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct-1M"  # Replace with your desired model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text
tokens = tokenizer(text)['input_ids']
print(f"Number of tokens: {len(tokens)}")

if limit_context_tokens is not None:
    if len(tokens) > limit_context_tokens:
        tokens = tokens[:limit_context_tokens]
        print(f"Tokens truncated to {limit_context_tokens} tokens")
        text = tokenizer.decode(tokens)

list_of_questions = [
    "Summarize the text in few sentences.",
    "What are the main points discussed in the text?",
    "What is the main theme of the text?",
    "What are the key arguments presented in the text?",
    "Who is the main character in the text?",
    "Describe shortly the main character.",
    "What was the most funny part of the text?",
    "What was the most sad part of the text?",
    "What was the most interesting part of the text?",
    "Summarize shortly the first paragraph of the text.",
]
dataset = ""
for question in list_of_questions:
    prompt = f"For the given CONTEXT answer the QUESTION. \n CONTEXT: {text}\n QUESTION {question}\n"
    item = {"prompt": prompt }
    dataset += dataset + json.dumps(item, ensure_ascii=False) + "\n"


# Save the dataset to a JSON file
output_file = "dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(dataset)
print(f"Dataset saved to {output_file}")
