from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import openvino as ov
import torch
import os

# Replace this with your LAION model
model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
print(f"Downloading pretrained model: {model_id}")

# Load processor and model
processor = CLIPProcessor.from_pretrained(model_id)
full_model = CLIPModel.from_pretrained(model_id)
image_encoder = full_model.vision_model
image_encoder.eval()

# Dummy image input for tracing
image = Image.new("RGB", (224, 224))
inputs = processor(images=image, return_tensors="pt")["pixel_values"]

# Convert to OpenVINO IR
print("Converting image encoder to OpenVINO IR...")
ov_model = ov.convert_model(image_encoder, example_input=inputs)
ov.save_model(ov_model, "clip_image_encoder.xml")
print("Model saved!")

# Move to proper OVMS path
mod_path = "saved_mod/laion/1"
os.makedirs(mod_path, exist_ok=True)
os.replace("clip_image_encoder.xml", f"{mod_path}/clip_image_encoder.xml")
os.replace("clip_image_encoder.bin", f"{mod_path}/clip_image_encoder.bin")
print(f"Model ready at {mod_path} for OpenVINO Model Server")
