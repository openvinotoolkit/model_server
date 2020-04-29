import json

import pytest
import requests
from marshmallow import ValidationError

from ams_schemas import InferenceResponseSchema
from fixtures.ams_fixtures import small_object_detection_image, \
     medium_object_detection_image, large_object_detection_image, \
     png_object_detection_image, jpg_object_detection_image, bmp_object_detection_image


def validate_ams_inference_response_schema(response: dict):
    try:
        parsed_response = InferenceResponseSchema().validate(response)
    except ValidationError as e:
        print('Response {} has invalid schema.'.format(response))
        print(e)
        raise


class TestAmsWrapper:
    def test_empty_input(self, ams_object_detection_model_endpoint):
        wrong_input = b''
        response = requests.post(ams_object_detection_model_endpoint,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(wrong_input))},
                                 data=wrong_input)
        assert response.status_code == 400

    # def test_wrong_input_image(self, ams_object_detection_model_endpoint):
    #     wrong_input = b'BLABLABLA'
    #     # TODO: define User-Agent header?
    #     response = requests.post(ams_object_detection_model_endpoint,
    #                              headers={'Content-Type': 'image/png',
    #                                       'Content-Length': str(len(wrong_input))},
    #                              body=wrong_input)
    #     assert response.status_code == 400

    def test_wrong_input_content_type(self, ams_object_detection_model_endpoint):
        response = requests.post(ams_object_detection_model_endpoint,
                                 headers={'Content-Type': 'bad-content-type'},
                                 data=b'some_data')
        assert response.status_code == 400

    @pytest.mark.parametrize("image", [small_object_detection_image(),
                                       medium_object_detection_image(),
                                       large_object_detection_image()])
    def test_input_image_different_sizes(self, ams_object_detection_model_endpoint, image):
        response = requests.post(ams_object_detection_model_endpoint,
                                 headers={'Content-Type': 'image/png',
                                          'Content-Length': str(len(image))},
                                 data=image)

        assert response.status_code == 200
        assert response.headers.get('Content-Type') == 'application/json'

        response_json = response.json()

        assert response_json.get('inferences')
        for inference_response in response_json['inferences']:
            validate_ams_inference_response_schema(inference_response)

    @pytest.mark.parametrize("image_format,image", [('image/png', png_object_detection_image()),
                                                    ('image/jpg', jpg_object_detection_image()),
                                                    ('image/bmp', bmp_object_detection_image())])
    def test_input_image_different_formats(self, ams_object_detection_model_endpoint,
                                           image_format, image):
        response = requests.post(ams_object_detection_model_endpoint,
                                 headers={'Content-Type': image_format,
                                          'Content-Length': str(len(image))},
                                 data=image)
        
        assert response.status_code == 200
        assert response.headers.get('Content-Type') == 'application/json'

        response_json = response.json()

        assert response_json.get('inferences')
        for inference_response in response_json['inferences']:
            validate_ams_inference_response_schema(inference_response)

    # @pytest.mark.parametrize("image,expected_instances", [(object_detection_image_no_entity, 0),
    #                                                       (object_detection_image_one_entity, 1),
    #                                                       (object_detection_image_five_entities, 5)])
    # def test_object_detection_entity(self, ams_object_detection_model_endpoint,
    #                                  image, expected_instances):
    #     response = requests.post(ams_object_detection_model_endpoint,
    #                              headers={'Content-Type': 'image/png',
    #                                       'Content-Length': str(len(image))},
    #                              body=image)
        
    #     assert response.status_code == 200
    #     assert response.headers.get('Content-Type') == 'application/json'
    #     assert response.get('inferences') and len(response.get('inferences'))  == expected_instances
    #     for inference_response in response['inferences']:
    #         validate_ams_inference_response_schema(inference_response)
    
