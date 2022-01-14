import argparse
from ovmsclient import make_grpc_client, make_grpc_predict_request
from utils.common import get_model_io_names, read_image_paths, read_imgs_as_ndarray, get_model_input_shape
from utils.ssd_mobilenet_utils import ssd_mobilenet_postprocess


# read .txt file with listed images into ndarray in model shape
def read_imgs_as_ndarray(images_dir, shape, layout):
    paths = [join(images_dir, f) for f in listdir(images_dir) if isfile(join(images_dir, f))]
    imgs = np.zeros([0, shape[1], shape[2], shape[3]])
    for path in paths:
        path = path.strip()
        img = getJpeg(path, shape, layout)
        imgs = np.append(imgs, img, axis=0)
    return imgs.astype('float32')


def crop_resize(img,cropx,cropy,layout):
    y, x, c = img.shape
    if y < cropy:
        img = cv2.resize(img, (x, cropy))
        y = cropy
    if x < cropx:
        img = cv2.resize(img, (cropx, y))
        x = cropx
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def getJpeg(path, shape, layout, rgb_image=0):
    with open(path, mode='rb') as file:
        content = file.read()

    img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR format
    # retrived array has BGR format and 0-255 normalization
    # format of data is HWC
    # add image preprocessing if needed by the model
    if layout == "NCHW":
        img = crop_resize(img, shape[3], shape[2], layout)
    else:   #layout == "NHWC"
        img = crop_resize(img, shape[2], shape[1], layout)
    img = img.astype('float32')
    #convert to RGB instead of BGR if required by model
    if rgb_image:
        img = img[:, :, [2, 1, 0]]
    # switch from HWC to CHW and reshape to 1,3,size,size for model blob input requirements
    if layout == "NCHW":
        img = img.transpose(2, 0, 1)
    return img.reshape(shape)


parser = argparse.ArgumentParser(description='Make vehicle detection prediction using images in binary format')
parser.add_argument('--images_dir', required=True,
                    help='Path to a directory with images in JPG or PNG format')
parser.add_argument('--grpc_address', required=False, default='localhost',
                    help='Specify url to grpc service. default:localhost')
parser.add_argument('--grpc_port', required=False, default=9000, type=int,
                    help='Specify port to grpc service. default: 9000')
parser.add_argument('--model_name', default='vehicle-detection',
                    help='Model name to query. default: vehicle-detection')
parser.add_argument('--model_version', default=0, type=int,
                    help='Model version to query. default: latest available')
parser.add_argument('--output_save_path', required=True,
                    help='Path to store output.')
args = vars(parser.parse_args())

# configuration
images_dir = args.get('images_dir')
address = args.get('grpc_address')
port = args.get('grpc_port')
model_name = args.get('model_name')
model_version = args.get('model_version')
output_save_path = args.get('output_save_path')

# creating grpc client
config = {
    "address": address,
    "port": port
}
client = make_grpc_client(config)

# receiving metadata from model
input_name, output_name = get_model_io_names(client, model_name, model_version)
input_shape = get_model_input_shape(client, model_name, model_version)
input_layout = "NCHW"

# preparing images
imgs = read_imgs_as_ndarray(images_dir, input_shape, input_layout)
imgs_paths = read_image_paths(images_dir)

for i, img in enumerate(imgs):
    img_path = imgs_paths[i]
    # preparing predict request
    inputs = {
        input_name: [img]
    }
    request = make_grpc_predict_request(inputs, model_name, model_version)

    # sending predict request and receiving response
    response = client.predict(request)

    ssd_mobilenet_postprocess(response, img_path, output_name, output_save_path)
