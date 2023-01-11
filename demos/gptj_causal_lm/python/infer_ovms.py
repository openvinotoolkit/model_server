import ovmsclient
import torch
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Demo for simple inference to GPT-J-6b model')
parser.add_argument('--url', required=False, help='Url to connect to', default='localhost:9000')
parser.add_argument('--model_name', required=False, help='Model name in the serving', default='gpt-j-6b')
args = vars(parser.parse_args())

client = ovmsclient.make_grpc_client(args['url'])
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = "OpenVINO Model Server is"
inputs = tokenizer(input_sentence, return_tensors="np")
results = client.predict(inputs=dict(inputs), model_name=args['model_name'])

predicted_token_id = token = torch.argmax(torch.nn.functional.softmax(torch.Tensor(results[0,-1,:]),dim=-1),dim=-1)
word = tokenizer.decode(predicted_token_id)

print(results)
print('predicted word:', word)
