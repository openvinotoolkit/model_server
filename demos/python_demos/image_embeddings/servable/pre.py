from pyovms import Tensor
from transformers import CLIPProcessor,AutoImageProcessor
from PIL import Image
import numpy as np
from io import BytesIO
from tritonclient.utils import deserialize_bytes_tensor

class OvmsPythonModel:
    def initialize(self, kwargs:dict):
        self.node_name=kwargs.get("node_name","")

        if "clip" in self.node_name.lower():
            self.processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.mode="clip"

        elif "dino" in self.node_name.lower():
            self.processor=AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.mode="dino"

        elif "laion" in self.node_name.lower():
            self.processor=CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
            self.mode="laion"
        else:
            raise ValueError(f"Unsupported model type in node name: {self.node_name}")

    def execute(self,inputs: list):
        image_bytes = deserialize_bytes_tensor(bytes(inputs[0]))[0]

        image=Image.open(BytesIO(image_bytes)).convert("RGB")
        processed=self.processor(images=image,return_tensors="np")
        pixel_values=processed["pixel_values"].astype(np.float32)
        return[Tensor(name="pixel_values",buffer=pixel_values)]

