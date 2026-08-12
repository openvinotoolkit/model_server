import glob
import json
import os
import requests
import openvino as ov
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

COCO_CLASSES_URL = "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_80cl.txt"

MODEL_REPO = "OpenVINO/yolox_tiny-fp16-ov"
# ---------------------------------------------------------
# 1. Download model
# ---------------------------------------------------------

model_dir = snapshot_download(repo_id=MODEL_REPO)

xml_path = glob.glob(os.path.join(model_dir, "*.xml"))[0]
config_path = os.path.join(model_dir, "config.json")

print("Found IR    :", xml_path)
print("Found config:", config_path)


# ---------------------------------------------------------
# 2. Read config.json
# ---------------------------------------------------------

with open(config_path, "r") as f:
    config = json.load(f)

print("\nModel config:")
print("model_name :", config.get("model_name"))
print("model_type :", config.get("model_type"))
print("input_type :", config.get("input_dtype"))
print("mean_values:", config.get("mean_values"))
print("scale_values:", config.get("scale_values"))


# ---------------------------------------------------------
# 3. Parse mean and scale values
# ---------------------------------------------------------
mean_values = [float(x) for x in config["mean_values"].split()]
scale_values = [float(x) for x in config["scale_values"].split()]

print("\nParsed preprocessing:")
print("mean :", mean_values)
print("scale:", scale_values)

# ---------------------------------------------------------
# 4. Load OpenVINO model
# ---------------------------------------------------------
core = ov.Core()
model = core.read_model(xml_path)
# ---------------------------------------------------------
# 5. Configure preprocessing
# ---------------------------------------------------------
ppp = ov.preprocess.PrePostProcessor(model)
inp = ppp.input(0)

# Input coming from user/image:
# f32 NHWC
inp.tensor().set_element_type(ov.Type.f32) .set_layout(ov.Layout("NHWC"))

# Model expects:
# float32 NCHW
inp.model().set_layout(ov.Layout("NCHW"))

# Preprocessing:

inp.preprocess().convert_element_type(ov.Type.f32).convert_layout(ov.Layout("NCHW")).scale(255.0).mean(mean_values).scale(scale_values)
# ---------------------------------------------------------
# 6. Build and save
# ---------------------------------------------------------

model = ppp.build()

output_path = "yolox_tiny_float32/1/yolox_tiny_full_preprocess.xml"

ov.save_model(model, output_path)

print("\nSaved:", os.path.abspath(output_path))

with open("coco_80cl.txt", "wb") as f:
    f.write(requests.get(COCO_CLASSES_URL).content)

print("Downloaded successfully")