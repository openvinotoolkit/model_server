import os.path
import urllib.request
import tarfile

mobilenet_url="https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar"
mobilenetv3_model_path = "model/MobileNetV3_large_x1_0_infer/inference.pdmodel"
if os.path.isfile(mobilenetv3_model_path): 
    print("Model MobileNetV3_large_x1_0 already existed")
else:
    #Download the model from the server, and untar it.
    print("Downloading the MobileNetV3_large_x1_0_infer model (20Mb)... May take a while...")
    #make the directory if it is not 
    os.makedirs('model')
    urllib.request.urlretrieve(mobilenet_url, "model/MobileNetV3_large_x1_0_infer.tar")
    print("Model Downloaded")

    file = tarfile.open("model/MobileNetV3_large_x1_0_infer.tar")
    res = file.extractall('model')
    file.close()
    if (not res):
        print("Model Extracted to \"model/MobileNetV3_large_x1_0_infer\".")
    else:
        print("Error Extracting the model. Please check the network.")
