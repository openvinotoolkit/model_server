# This script generates a dataset of long context examples for training and evaluation.
import os
import json
import requests
from transformers import AutoTokenizer
from openai import OpenAI
import time

client = OpenAI(
  base_url="http://ov-lnl-01.sclab.intel.com:8000/v3",
  api_key="unused",
  timeout=1800,
)
model = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model)

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
    first_chunk_time = time.time()
    response = client.completions.create(
         model=model,
         prompt=prompt,
         temperature=0.0,
         max_tokens=50,
         stream=True
    )
    second_chunk_time = None
    answer = ""
    for chunk in response:
        if second_chunk_time is None and hasattr(chunk.choices[0], "text") and chunk.choices[0].text != "":
            second_chunk_time = time.time()
            answer += chunk.choices[0].text
    ttft = (second_chunk_time - first_chunk_time)*1000 if first_chunk_time else None
    return answer, ttft


def get_trucated_text(text, limit_tokens=3000):
    tokens = tokenizer(text)['input_ids']
    if len(tokens) > limit_tokens:
        tokens = tokens[:limit_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


url = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/2011/donquix-2011.txt"

text = download_file(url)

lenghts = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
for l in lenghts:
    #print(f"Processing text with length limit: {l}")
    text_short = get_trucated_text(text, limit_tokens=l)
    #print(f"Truncated text length: {len(text_short)}")
    answer, ttft = get_answer(text_short)
    print("lenght",l,"TTFT", ttft)  # Print first 100 characters of the answer
