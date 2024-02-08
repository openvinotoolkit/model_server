#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from ovmsclient import make_grpc_client
import cv2
import numpy as np
import argparse
import random
from typing import Optional, Dict

parser = argparse.ArgumentParser(description='Make object detection prediction using images in binary format')
parser.add_argument('--image', required=True,
                    help='Path to a image in JPG or PNG format')
parser.add_argument('--service_url', required=False, default='localhost:9000',
                    help='Specify url to grpc service. default:localhost:9000', dest='service_url')
parser.add_argument('--model_name', default='faster_rcnn',
                    help='Model name to query. default: faster_rcnn')
parser.add_argument('--input_name', default='input_tensor',
                    help='Input name to query. default: input_tensor')
parser.add_argument('--model_version', default=0, type=int,
                    help='Model version to query. default: latest available')
parser.add_argument('--labels', default="coco_91cl.txt", type=str,
                    help='Path to COCO dataset labels with human readable class names')
parser.add_argument('--output', required=True,
                    help='Path to store output.')
args = parser.parse_args()
image = cv2.imread(filename=str(args.image))
image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(src=image, dsize=(255, 255))
network_input_image = np.expand_dims(resized_image, 0)

client = make_grpc_client(args.service_url)
inputs = {
    args.input_name: network_input_image
}

response = client.predict(inputs, args.model_name, args.model_version)

def add_detection_box(box, image, label):
    """
    Helper function for adding single bounding box to the image

    Parameters
    ----------
    box : np.ndarray
        Bounding box coordinates in format [ymin, xmin, ymax, xmax]
    image : np.ndarray
        The image to which detection box is added
    label : str, optional
        Detection box label string, if not provided will not be added to result image (default is None)

    Returns
    -------
    np.ndarray
        NumPy array including both image and detection box

    """
    ymin, xmin, ymax, xmax = box
    point1, point2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
    box_color = [random.randint(0, 255) for _ in range(3)] # nosec
    line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    cv2.rectangle(img=image, pt1=point1, pt2=point2, color=box_color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        font_face = 0
        font_scale = line_thickness / 3
        font_color = (255, 255, 255)
        text_size = cv2.getTextSize(text=label, fontFace=font_face, fontScale=font_scale, thickness=font_thickness)[0]
        # Calculate rectangle coordinates
        rectangle_point1 = point1
        rectangle_point2 = (point1[0] + text_size[0], point1[1] - text_size[1] - 3)
        # Add filled rectangle
        cv2.rectangle(img=image, pt1=rectangle_point1, pt2=rectangle_point2, color=box_color, thickness=-1, lineType=cv2.LINE_AA)
        # Calculate text position
        text_position = point1[0], point1[1] - 3
        # Add text with label to filled rectangle
        cv2.putText(img=image, text=label, org=text_position, fontFace=font_face, fontScale=font_scale, color=font_color, thickness=font_thickness, lineType=cv2.LINE_AA)
    return image

def visualize_inference_result(inference_result, image: np.ndarray, labels_map: Dict, detections_limit: Optional[int] = None):
    """
    Helper function for visualizing inference result on the image

    Parameters
    ----------
    inference_result : OVDict
        Result of the compiled model inference on the test image
    image : np.ndarray
        Original image to use for visualization
    labels_map : Dict
        Dictionary with mappings of detection classes numbers and its names
    detections_limit : int, optional
        Number of detections to show on the image, if not provided all detections will be shown (default is None)
    """
    detection_boxes: np.ndarray = inference_result["detection_boxes"]
    detection_classes: np.ndarray = inference_result["detection_classes"]
    detection_scores: np.ndarray = inference_result["detection_scores"]
    num_detections: np.ndarray = inference_result["num_detections"]

    detections_limit = int(
        min(detections_limit, num_detections[0])
        if detections_limit is not None
        else num_detections[0]
    )

    # Normalize detection boxes coordinates to original image size
    original_image_height, original_image_width, _ = image.shape
    normalized_detection_boxex = detection_boxes[::] * [
        original_image_height,
        original_image_width,
        original_image_height,
        original_image_width,
    ]

    image_with_detection_boxex = np.copy(image)

    for i in range(detections_limit):
        detected_class_name = labels_map[int(detection_classes[0, i])]
        score = detection_scores[0, i]
        label = f"{detected_class_name} {score:.2f}"
        add_detection_box(
            box=normalized_detection_boxex[0, i],
            image=image_with_detection_boxex,
            label=label,
        )
    return image_with_detection_boxex

with open(args.labels, "r") as file:
    coco_labels = file.read().strip().split("\n")
    coco_labels_map = dict(enumerate(coco_labels, 1))
    img = visualize_inference_result(response, image, coco_labels_map, 10)
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output, img)
