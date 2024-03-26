from openai import OpenAI
client = OpenAI(api_key="_", base_url="http://localhost:9098/v1")
#client = OpenAI(api_key="_", base_url="http://localhost:11338/v1")

completion = client.chat.completions.create(
  model="python_model",
  messages=[
    {"role": "user", "content": "What is OpenVINO?"},
  ],
  stream=True
)

for chunk in completion:
  print(chunk.choices[0].delta.content, flush=True, end='')
print()
