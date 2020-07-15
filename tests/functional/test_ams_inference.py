#
# Copyright (c) 2020 Intel Corporation
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

import pytest
import requests
from marshmallow import ValidationError

from ams_schemas import InferenceResponseSchema
from fixtures.ams_fixtures import small_object_detection_image, \
    medium_object_detection_image, large_object_detection_image, \
    png_object_detection_image, jpg_object_detection_image, \
    bmp_object_detection_image


def validate_inference_response_schema(response: dict):
    try:
        InferenceResponseSchema().validate(response)
    except ValidationError as e:
        print('Response {} has invalid schema.'.format(response))
        print(e)
        raise


class TestAmsInference:
    def test_empty_input(self, start_ams_service):
        _, ports = start_ams_service
        ams_port = ports['port']
        target = "vehicleDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
        wrong_input = b''
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(wrong_input))},
                                 data=wrong_input)
        assert response.status_code == 400

    def test_wrong_input_image(self, start_ams_service):
        _, ports = start_ams_service
        ams_port = ports['port']
        target = "vehicleDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
        wrong_input = b'INVALIDINPUT'
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(wrong_input))},
                                 data=wrong_input)
        assert response.status_code == 400

    def test_wrong_model_name(self, start_ams_service, jpg_object_detection_image):
        _, ports = start_ams_service
        ams_port = ports['port']
        target = "noSuchModelDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
        with open(jpg_object_detection_image, mode='rb') as image_file:
            image_bytes = image_file.read()
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(jpg_object_detection_image))},
                                 data=image_bytes)
        assert response.status_code == 404

    def test_wrong_input_content_type(self, start_ams_service):
        _, ports = start_ams_service
        ams_port = ports['port']
        target = "vehicleDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)

        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'bad-content-type'},
                                 data=b'some_data')
        assert response.status_code == 400

    @pytest.mark.parametrize("image", [small_object_detection_image(),
                                       medium_object_detection_image(),
                                       large_object_detection_image()])
    def test_input_image_different_sizes(self, start_ams_service, image):
        with open(image, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service
        ams_port = ports['port']
        targets = ["vehicleDetection", "vehicleClassification",
                   "personVehicleBikeDetection", "faceDetection"]
        for target in targets:
            endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
            response = requests.post(endpoint_url,
                                     headers={'Content-Type': 'image/png',
                                              'Content-Length': str(len(image))},
                                     data=image_bytes)
            print(response.text)
            assert (response.status_code == 200 and response.headers.get('Content-Type') == 'application/json') or \
                   (response.status_code == 204 and response.headers.get('Content-Type') is None)

            if response.status_code == 200:
                response_json = response.json()
                validate_inference_response_schema(response_json)

    @pytest.mark.parametrize("image_format,image", [('image/png', png_object_detection_image()),
                                                    ('image/jpg',
                                                     jpg_object_detection_image()),
                                                    ('image/bmp', bmp_object_detection_image())])
    def test_input_image_different_formats(self, start_ams_service, image_format, image):
        with open(image, mode='rb') as image_file:
            image_bytes = image_file.read()

        _, ports = start_ams_service
        ams_port = ports['port']
        targets = ["vehicleDetection", "vehicleClassification",
                   "personVehicleBikeDetection", "faceDetection"]
        for target in targets:
            endpoint_url = "http://localhost:{}/{}".format(ams_port, target)

            response = requests.post(endpoint_url,
                                     headers={'Content-Type': image_format,
                                              'Content-Length': str(len(image))},
                                     data=image_bytes)

            assert (response.status_code == 200 and response.headers.get('Content-Type') == 'application/json') or \
                   (response.status_code == 204 and response.headers.get('Content-Type') is None)

            if response.status_code == 200:
                response_json = response.json()
                validate_inference_response_schema(response_json)

    def test_input_blank_image(self, start_ams_service, object_detection_image_no_entities):
        with open(object_detection_image_no_entities, mode='rb') as image_file:
            image_bytes = image_file.read()

        _, ports = start_ams_service
        ams_port = ports['port']
        target = "vehicleDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(image_bytes))},
                                 data=image_bytes)
        assert response.status_code == 204

    def test_ams_without_ovms(self, start_ams_service_without_ovms, png_object_detection_image):
        with open(png_object_detection_image, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service_without_ovms
        ams_port = ports['port']
        target = "vehicleDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(image_bytes))},
                                 data=image_bytes)
        assert response.status_code == 503

    def test_ams_with_wrong_model_name(self, start_ams_service_with_wrong_model_name, jpg_object_detection_image):
        with open(jpg_object_detection_image, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service_with_wrong_model_name
        ams_port = ports['port']
        target = "vehicleDetection"
        endpoint_url = "http://localhost:{}/{}".format(ams_port, target)
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(image_bytes))},
                                 data=image_bytes)
        assert response.status_code == 500

    def test_vehicleClassification(self, start_ams_service, object_classification_red_truck):
        with open(object_classification_red_truck, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service
        ams_port = ports['port']

        endpoint_url = "http://localhost:{}/{}".format(
            ams_port, "vehicleClassification")
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(object_classification_red_truck))},
                                 data=image_bytes)
        assert response.status_code == 200
        assert response.headers.get('Content-Type') == 'application/json'

        response_json = response.json()
        print(response_json)
        #print("Hellooooooooooooooo")
        highest_probability = 0.0
        highest_value = ""
        attribute_name = ""

        highest_value=response_json["inferences"][1]["classification"]["tag"]

        #for classification in response_json["inferences"][1]:
        #            if classification["confidence"] > highest_probability:
        #                highest_probability = classification["confidence"]
        #                highest_value = classification["value"]
        #                attribute_name = classification["name"]
        #print("********************")
        #print(classification)
        #highest_value=classification[0]['tag']

        #        assert highest_probability > 0.547
        assert highest_value == "red"
        #        assert attribute_name == "color"

        highest_probability = 0.0
        highest_value = ""
        attribute_name = ""

        highest_value=response_json["inferences"][0]["classification"]["tag"]

        #        for classification in response_json["inferences"][0]["classification"]:
        #            if classification["confidence"] > highest_probability:
        #                highest_probability = classification["confidence"]
        #                highest_value = classification["value"]
        #                attribute_name = classification["name"]
        #            print(classification)
        #            highest_value=classification['tag']

        #        assert highest_probability > 0.9421
        assert highest_value == "truck"
    #        assert attribute_name == "type"

    def test_vehicleDetection(self, start_ams_service, object_detection_image_two_entities):
        with open(object_detection_image_two_entities, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service
        ams_port = ports['port']

        endpoint_url = "http://localhost:{}/{}".format(
            ams_port, "vehicleDetection")
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(object_detection_image_two_entities))},
                                 data=image_bytes)
        assert response.status_code == 200
        assert response.headers.get('Content-Type') == 'application/json'

        response_json = response.json()

        highest_probability = 0.0
        highest_box = dict()
        tag_value = ""
        detections_count = 0

        for detection in response_json["entities"]:
            detections_count += 1

            if detection["tag"]["confidence"] > highest_probability:
                highest_probability = detection["tag"]["confidence"]
                highest_box = detection["box"]
                tag_value = detection["tag"]["value"]

        epsilon = 0.000001
        assert highest_probability > 0.67
        assert tag_value == "vehicle"
        assert detections_count == 1
        assert 0.034460186958313 - \
               epsilon <= highest_box["w"] <= 0.034460186958313 + epsilon
        assert 0.0431380569934845 - \
               epsilon <= highest_box["h"] <= 0.0431380569934845 + epsilon
        assert 0.783527314662933 - \
               epsilon <= highest_box["l"] <= 0.783527314662933 + epsilon
        assert 0.173053205013275 - \
               epsilon <= highest_box["t"] <= 0.173053205013275 + epsilon

    def test_faceDetection(self, start_ams_service, object_detection_image_one_entity):
        with open(object_detection_image_one_entity, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service
        ams_port = ports['port']

        endpoint_url = "http://localhost:{}/{}".format(
            ams_port, "faceDetection")
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(object_detection_image_one_entity))},
                                 data=image_bytes)
        assert response.status_code == 200
        assert response.headers.get('Content-Type') == 'application/json'

        response_json = response.json()

        detection = max(
            response_json['entities'], key=lambda entity: entity['tag']['confidence'])

        assert detection['tag']['confidence'] >= 0.98
        assert detection['tag']['value'] == 'face'

        epsilon = 0.000001

        assert 0.38243523240089417 - \
               epsilon <= detection['box']['l'] <= 0.38243523240089417 + epsilon
        assert 0.28849169611930847 - \
               epsilon <= detection['box']['t'] <= 0.28849169611930847 + epsilon
        assert 0.036220431327819824 - \
               epsilon <= detection['box']['w'] <= 0.036220431327819824 + epsilon
        assert 0.07158094644546509 - \
               epsilon <= detection['box']['h'] <= 0.07158094644546509 + epsilon

    def test_personVehicleBikeDetection(self, start_ams_service, object_detection_image_one_entity):
        with open(object_detection_image_one_entity, mode='rb') as image_file:
            image_bytes = image_file.read()
        _, ports = start_ams_service
        ams_port = ports['port']

        endpoint_url = "http://localhost:{}/{}".format(
            ams_port, "personVehicleBikeDetection")
        response = requests.post(endpoint_url,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(object_detection_image_one_entity))},
                                 data=image_bytes)
        assert response.status_code == 200
        assert response.headers.get('Content-Type') == 'application/json'

        response_json = response.json()

        detection = max(
            response_json['entities'], key=lambda entity: entity['tag']['confidence'])

        assert detection['tag']['confidence'] >= 0.99
        assert detection['tag']['value'] == 'pedestrian'

        epsilon = 0.000001

        assert 0.3313550353050232 - \
               epsilon <= detection['box']['l'] <= 0.3313550353050232 + epsilon
        assert 0.25375649333000183 - \
               epsilon <= detection['box']['t'] <= 0.25375649333000183 + epsilon
        assert 0.1486881971359253 - \
               epsilon <= detection['box']['w'] <= 0.1486881971359253 + epsilon
        assert 0.7409175932407379 - \
               epsilon <= detection['box']['h'] <= 0.7409175932407379 + epsilon

    # @pytest.mark.parametrize("image,expected_instances", [(object_detection_image_no_entity, 0),
    #                                                       (object_detection_image_one_entity, 1),
    #                                                       (object_detection_image_two_entities, 2)])
    # def test_object_detection_entity(self, ams_object_detection_model_endpoint,
    #                                  image, expected_instances):
    #     response = requests.post(ams_object_detection_model_endpoint,
    #                              headers={'Content-Type': 'image/png',
    #                                       'Content-Length': str(len(image))},
    #                              body=image)

    #     assert response.status_code == 200
    #     assert response.headers.get('Content-Type') == 'application/json'
    #     assert response.get('inferences') and \
    #            len(response.get('inferences', {}).get('entities'))  == expected_instances
    #     for inference_response in response['inferences']:
    #         validate_ams_inference_response_schema(inference_response)
