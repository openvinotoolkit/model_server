import requests
import base64
import argparse
base_url='http://ov-spr-28.sclab.intel.com:8005/v3'
#model_name = "OpenGVLab/InternVL2_5-8B"
model_name = "OpenGVLab/InternVL2-2B"
parser = argparse.ArgumentParser(description="Provide the image file path.")
parser.add_argument("image", type=str, help="Path to the image file.")
args = parser.parse_args()

image = args.image
#image2 = args.image2

from openai import OpenAI
import time
import argparse
client = OpenAI(api_key='unused', base_url=base_url, timeout=17800)

def convert_image(Image):
    with open(Image,'rb' ) as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return base64_image

start_time = time.time()
stream = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
              {"type": "text", "text": "Transcribe the text."},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{convert_image(image)}"}},
            ]
        }
        ],
    temperature=0.0,
    max_tokens=400,
    stream=True,
)

for chunk in stream:
    end_time = time.time()
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    #print(f"\nTime taken to get this chunk: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    



# def convert_image(Image):
#     with open(Image,'rb' ) as file:
#         base64_image = base64.b64encode(file.read()).decode("utf-8")
#     return base64_image

# import requests
# payload = {"model": "OpenGVLab/InternVL2_5-8B", 
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#               {"type": "text", "text": "Describe what is one the picture."},
#               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{convert_image('zebra.jpeg')}"}}
#             ]
#         }
#         ],
#     "max_completion_tokens": 100
# }
# headers = {"Content-Type": "application/json", "Authorization": "not used"}
# response = requests.post(base_url + "/chat/completions", json=payload, headers=headers)
# print(response.text)