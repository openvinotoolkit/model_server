from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import openvino as ov
import os

model_id = "facebook/dinov2-base"
print(f"Downloading pretrained model {model_id}...")

model = AutoModel.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)

image = Image.new("RGB", (224, 224))
inputs = processor(images=image, return_tensors="pt")["pixel_values"]

print("Converting models...")
ov_model = ov.convert_model(model, example_input=inputs)
ov.save_model(ov_model, "dino_image_encoder.xml")
print("Model saved!")

mod_path = "saved_mod/dino/1"
os.makedirs(mod_path, exist_ok=True)
os.replace("dino_image_encoder.xml", f"{mod_path}/dino_image_encoder.xml")
os.replace("dino_image_encoder.bin", f"{mod_path}/dino_image_encoder.bin")
print("Model ready for OVMS")



