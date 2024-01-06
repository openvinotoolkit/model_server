#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import cv2
import numpy as np

from google.cloud import storage
from use_cases.use_case import UseCase
from datetime import datetime, MINYEAR


class PersonVehicleBikeDetection(UseCase):
    CLASSES = ["Vehicle", "Person", "Bike"]
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    CONFIDENCE_THRESHOLD = float(os.environ.get('PERSON_DETECTION_CONFIDENCE_THRESHOLD', 0.75))
    frame: np.ndarray = None

    def supports_visualization() -> bool:
        return True

    def visualize(frame: np.ndarray, inference_result: np.ndarray) -> np.ndarray:

        # expecting inference_result in shape (1, 1, 200, 7)
        # ommiting batch dimension - frames are not analyzed in batches
        inference_result = inference_result[0][0]
        for single_prediction in inference_result:
            img_id = single_prediction[0]
            label = single_prediction[1]
            conf = single_prediction[2]
            x_min = single_prediction[3]
            y_min = single_prediction[4]
            x_max = single_prediction[5]
            y_max = single_prediction[6]

            if img_id != -1 and conf >= PersonVehicleBikeDetection.CONFIDENCE_THRESHOLD:
                height, width = frame.shape[0:2]
                cv2.rectangle(frame, (int(width * x_min), int(height * y_min)),
                              (int(width * x_max), int(height * y_max)), PersonVehicleBikeDetection.COLORS[int(label)],
                              1)
                cv2.putText(frame, str(PersonVehicleBikeDetection.CLASSES[int(label)]), (int(width * x_min) - 10,
                                                                                         int(height * y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            PersonVehicleBikeDetection.COLORS[int(label)], 1)
                PersonVehicleBikeDetection.frame = frame

        return frame

    def preprocess(frame: np.ndarray) -> np.ndarray:
        return frame

    def postprocess(inference_result: np.ndarray):
        """
        storage and log images of person detected
        :return:
        """
        inference_result = inference_result[0][0]
        for single_prediction in inference_result:
            img_id = single_prediction[0]
            label = single_prediction[1]
            conf = single_prediction[2]

            if img_id != -1 and conf >= PersonVehicleBikeDetection.CONFIDENCE_THRESHOLD:
                if str(PersonVehicleBikeDetection.CLASSES[int(label)]) == "Person":
                    # write the detected image to a file
                    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
                    filename = f"detected-person_{formatted_datetime}.png"
                    local_folder = os.environ.get('PERSON_DETECTION_LOCAL_FOLDER',
                                                  os.path.join(os.environ.get('HOME'), 'Pictures'))
                    if os.path.isdir(local_folder):
                        local_file = os.path.join(local_folder, filename)
                        cv2.imwrite(local_file, PersonVehicleBikeDetection.frame)
                    else:
                        local_file = None

                    # upload the file to Google Cloud Storage
                    if 'PERSON_DETECTION_GCS_BUCKET' in os.environ and 'PERSON_DETECTION_GCS_FOLDER' in os.environ:
                        gcs_folder_date_hour = (f"{formatted_datetime.split('_')[0]}/"
                                                f"{formatted_datetime.split('_')[1].split('-')[0]}")
                        blob_name = f"{os.environ.get('PERSON_DETECTION_GCS_FOLDER')}/{gcs_folder_date_hour}/{filename}"
                        gs_path = upload_blob(os.environ.get('PERSON_DETECTION_GCS_BUCKET'), local_file, blob_name)
                    else:
                        gs_path = None

                    sampled_log(
                        {
                            'confidence': float(conf),
                            'picture': local_file,
                            'uploaded_destination': gs_path,
                            'camera_id': os.environ.get('PERSON_DETECTION_CAMERA_ID', 'debug'),
                        }
                    )


# Throttle logging to avoid the same logging rate as FPS
last_logged_datetime = datetime(MINYEAR, 1, 1, 0, 0)
if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
    from google.cloud import logging

    logging_client = logging.Client()
    # This log can be found in the Cloud Logging console under resource type `Global`
    # logName: projects/$PROJECT_ID/logs/person-detection-debug
    gcp_logger = logging_client.logger(
        os.environ.get('PERSON_DETECTION_GCP_LOG_NAME', 'person-detection-debug')
    )
else:
    gcp_logger = None


def sampled_log(log_dict: dict):
    """Write logs to cloud logging and optionally the terminal

    :param log_dict: the dict must be serializable
    :return:
    """
    global last_logged_datetime
    min_log_interval_seconds = int(os.environ.get('PERSON_DETECTION_MIN_LOG_INTERVAL_SECONDS', 3))

    current_datetime = datetime.now()  # Calculate the current datetime
    # Calculate the time difference
    time_difference = current_datetime - last_logged_datetime

    if time_difference.total_seconds() > min_log_interval_seconds:
        log_dict['sample-interval-seconds'] = min_log_interval_seconds
        if 'PERSON_DETECTION_DEBUG' in os.environ:
            print(str(log_dict))
        if gcp_logger:
            gcp_logger.log_struct(log_dict, severity="INFO")
        last_logged_datetime = current_datetime


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.
    :param bucket_name: Google cloud storage bucket name
    :param source_file_name: /local/path/to/file.png
    :param destination_blob_name: folder-path/storage-object-name.png
    :return: gs://bucket_name/destination_blob_name
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)
    gs_path = f"gs://{bucket_name}/{destination_blob_name}"

    if 'PERSON_DETECTION_DEBUG' in os.environ:
        print(f"File {source_file_name} uploaded to {gs_path} ;")

    return gs_path
