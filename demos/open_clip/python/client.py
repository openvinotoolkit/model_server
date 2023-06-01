import ovmsclient
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import cv2
from scipy.special import softmax
from PIL import Image

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

# TODO: CLI
# TODO: --batch_size?
# TODO: path to single image param?

OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]

tokenizer = open_clip.get_tokenizer("ViT-B-16-plus-240")
client = ovmsclient.make_grpc_client("localhost:8913")

# Use preprocessing method from open_clip repo
#_, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained='laion400m_e32')

# Or re-create preprocessing manually using torchvision
def _convert_to_rgb(image):
    return image.convert('RGB')

preprocess = Compose([
    Resize(240, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(240),
    _convert_to_rgb,
    ToTensor(),
    Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
])

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

templates = [
    'a photo of a {c}.',
    'a blurry photo of a {c}.',
    'a black and white photo of a {c}.',
    'a low contrast photo of a {c}.',
    'a high contrast photo of a {c}.',
    'a bad photo of a {c}.',
    'a good photo of a {c}.',
    'a photo of a small {c}.',
    'a photo of a big {c}.',
    'a photo of the {c}.',
    'a blurry photo of the {c}.',
    'a black and white photo of the {c}.',
    'a low contrast photo of the {c}.',
    'a high contrast photo of the {c}.',
    'a bad photo of the {c}.',
    'a good photo of the {c}.',
    'a photo of the small {c}.',
    'a photo of the big {c}.'
]


def make_classifier():
    zeroshot_weights = []
    for classname in class_names:
        texts = [template.format(c=classname) for template in templates]
        tokens = tokenizer(texts).to("cpu").numpy().astype("int64")
        class_embeddings = client.predict(
            inputs={"input_ids": tokens},
            model_name="text_encoder"
        )
        class_embeddings = torch.from_numpy(class_embeddings)
        class_embeddings = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embeddings /= class_embeddings.norm()
        zeroshot_weights.append(class_embeddings)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


classifier = make_classifier()


def predict(image_path, label_path=None):
    print('processing', image_path, label_path)

    # preprocessing via open_clip repo (92.72% acc)
    img = preprocess(Image.open(image_path)).unsqueeze(0).numpy()

    # preprocessing via opencv (90.2% acc)
    # img = cv2.imread(image_path)
    # img_f = img.astype(np.float32)
    # img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
    # img_f = cv2.resize(img_f, (240,240), cv2.INTER_CUBIC)
    # img_f = img_f/np.array([255.0, 255.0, 255.0], dtype=np.float32)
    # img_f = (img_f - np.array(OPENAI_DATASET_MEAN, dtype=np.float32))/np.array(OPENAI_DATASET_STD, dtype=np.float32)
    # img_f = np.expand_dims(img_f, axis=0)
    # img = np.transpose(img_f, (0,3,1,2))


    image_features = client.predict(
        inputs={"image": img},
        model_name="image_encoder"
    )

    image_features = torch.from_numpy(image_features)
    image_features = F.normalize(image_features, dim=-1)

    logits = 100. * image_features @ classifier

    probs = softmax(logits, axis=1)
    lbs = np.array(class_names)[np.argsort(probs[0])]
    prbs = probs[0][np.argsort(probs[0])]
    for l, p in zip(reversed(lbs), reversed(prbs)):
        print(l, p)

    if label_path is None:
        return False

    with open(label_path) as f:
        expected = int(f.readline())

    return logits.argmax() == expected


samples = 200  # change this to 10000 to test the whole dataset
valid = 0
for i in range(0,samples):    
    success = predict(
        image_path="dataset/s" + str(i).rjust(7, '0') + ".webp",
        label_path="dataset/s" + str(i).rjust(7, '0') + ".cls")
    if success:
        valid += 1

print('accurracy', valid / samples * 100, '%')
