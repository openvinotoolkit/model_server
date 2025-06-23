from pyovms import Tensor
from transformers import CLIPProcessor
from PIL import Image
import numpy as np
from io import BytesIO
from tritonclient.utils import deserialize_bytes_tensor

class OvmsPythonModel:
    def initialize(self, kwargs:dict):
        model_id="openai/clip-vit-base-patch32"
        self.processor=CLIPProcessor.from_pretrained(model_id)

    def execute(self,inputs: list):
        image_bytes = deserialize_bytes_tensor(bytes(inputs[0]))[0]

        image=Image.open(BytesIO(image_bytes)).convert("RGB")
        processed=self.processor(images=image,return_tensors="np")
        pixel_values=processed["pixel_values"].astype(np.float32)
        return[Tensor(name="40",data=pixel_values)]

