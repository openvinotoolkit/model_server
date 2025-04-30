# This script generates a dataset of long context examples for training and evaluation.
import os
import json
import requests
from transformers import AutoTokenizer
from openai import OpenAI
import time

client = OpenAI(
  base_url="http://ov-spr-28.sclab.intel.com:8000/v3",
  api_key="unused",
  timeout=1800,
)

# function to download a file from a URL
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
        return None

def get_answer(prompt):
    start_time = time.time()
    print("Starting to get answer for prompt: {}:".format(prompt[-50:]))
    messages = [
            {"role": "system", "content": "You are a helpful assistant giving short answers."},
            {"role": "user", "content": prompt}
        ]
    print("messages:", messages)
    response = client.chat.completions.create(
        model="long_llm",
        messages=messages,
        temperature=0,
        max_completion_tokens=100,
        stream=True,
    )
    answer = ""
    for chunk in response:
        end_time = time.time()
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            answer += chunk.choices[0].delta.content
        print(f"\nTime taken to get this chunk: {end_time - start_time:.4f} seconds")
        start_time = time.time()
    print("\nAnswer received:{}".format(answer))
    return answer

# def get_answer(prompt):
#     response = client.completions.create(
#         model="Qwen/Qwen2.5-7B-Instruct-1M",
#         prompt="You are a helpful assistant. {}".format(prompt),
#         temperature=0
#     )
#     return response.choices[0].text
    
url = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2011/donquix-2011.txt"

text = download_file(url)

# Initialize the tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct-1M"  # Replace with your desired model name
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text
tokens = tokenizer(text)['input_ids']
print(f"Number of tokens: {len(tokens)}")

limit_tokens = 50000
if limit_tokens is not None:
    if len(tokens) > limit_tokens:
        tokens = tokens[:limit_tokens]
        print(f"Tokens truncated to {limit_tokens} tokens")
        text = tokenizer.decode(tokens)

# from openvino import compile_model
# from openvino_tokenizers import convert_tokenizer
# loaded_tokenizer = compile_model("/home/dtrawins/model_server/demos/common/export_models/models/Qwen/Qwen2.5-7B-Instruct-1M/openvino_tokenizer.xml")
# print(f"Loaded tokenizer:")
# loaded_ov_output = loaded_tokenizer([text])
# print(f"Number of OV tokens: {(loaded_ov_output['input_ids'].shape)}")

list_of_questions = [
    "Summarize the text in few sentences.",
    #"What are the main points discussed in the text?",
    # "What is the main theme of the text?",
    # "What are the key arguments presented in the text?",
    # "Who is the main character in the text?",
    # "Describe shortly the main character.",
    # "What was the most funny part of the text?",
    # "What was the most sad part of the text?",
    # "What was the most interesting part of the text?",
    # "Summarize shortly the first paragraph of the text.",
]
dataset = []
for question in list_of_questions:
    prompt = f"For the given CONTEXT answer the QUESTION. \n CONTEXT: {text}\n QUESTION {question}\n"
    answer = get_answer(prompt)
    conversation = [{
        "from": "human",
        "value": prompt
        },
        {
        "from": "gpt",
        "value": answer
    }
    ]
    item = {
        "id": len(dataset),
        "conversations": conversation,
    }
    dataset.append(item)


# Save the dataset to a JSON file
output_file = "dataset.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False, indent=4)
print(f"Dataset saved to {output_file}")
