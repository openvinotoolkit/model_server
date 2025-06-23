from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import openvino as ov
import os

model_id="openai/clip-vit-base-patch32"
print(f"Downloading pretrained model {model_id}...")

full_model=CLIPModel.from_pretrained(model_id)
model=full_model.vision_model
processor=CLIPProcessor.from_pretrained(model_id)

image=Image.new("RGB",(224,224))
inputs=processor(images=image,return_tensors="pt")["pixel_values"]

print("Converting model...")
ov_model=ov.convert_model(model,example_input=inputs)
ov.save_model(ov_model,"clip_image_encoder.xml")
print("Model saved!")

mod_path="saved_mod/1"
os.makedirs(mod_path,exist_ok=True)
os.replace("clip_image_encoder.xml", f"{mod_path}/clip_image_encoder.xml")
os.replace("clip_image_encoder.bin", f"{mod_path}/clip_image_encoder.bin")
print("Model ready for OVMS")




