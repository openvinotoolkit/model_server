import os
import time
import ovmsclient
import torch
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Demo for GPT-J causal LM requests using ovmsclient gRPC API')

parser.add_argument('--input', required=True, help='Beginning of a sentence', type=str)
args = vars(parser.parse_args())

client = ovmsclient.make_grpc_client("localhost:11340")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = args['input']

iteration = 0
while True:
    inputs = tokenizer(input_sentence, return_tensors="np")
    start_time = time.time()
    results = client.predict(inputs=dict(inputs), model_name='resnet')
    last_latency = time.time() - start_time
    predicted_token_id = token = torch.argmax(torch.nn.functional.softmax(torch.Tensor(results[0,-1,:]),dim=-1),dim=-1)
    word = tokenizer.decode(predicted_token_id)
    input_sentence += word
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Iteration: {iteration}\nLast predicted token: {predicted_token_id}\nLast latency: {last_latency}s\n{input_sentence}")
    iteration += 1
    if predicted_token_id == 198:
        break

