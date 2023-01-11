import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

input_sentence = "OpenVINO Model Server is"
inputs = tokenizer(input_sentence, return_tensors="pt")

outputs = model(**inputs, labels=inputs["input_ids"])
predicted_token_id = torch.argmax(torch.nn.Softmax(dim=-1)(outputs.logits[0,-1,:]))
word = tokenizer.decode(predicted_token_id)

print(outputs.logits)
print('predicted word:', word)
