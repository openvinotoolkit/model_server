#
# Copyright (c) 2019 Intel Corporation
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

import datetime
import json

import falcon
from google.protobuf.json_format import MessageToJson
from tensorflow_serving.apis import get_model_metadata_pb2, \
    get_model_status_pb2

from ie_serving.logger import get_logger
from ie_serving.server.constants import WRONG_MODEL_SPEC, INVALID_FORMAT, \
    OUTPUT_REPRESENTATION, REST
from ie_serving.server.get_model_metadata_utils import \
    prepare_get_metadata_output
from ie_serving.server.predict_utils import prepare_input_data, statusCodes
from ie_serving.server.request import Request
from ie_serving.server.rest_msg_processing import preprocess_json_request, \
    prepare_json_response
from ie_serving.server.rest_msg_validation import get_input_format
from ie_serving.server.service_utils import \
    check_availability_of_requested_model, \
    check_availability_of_requested_status, add_status_to_response

logger = get_logger(__name__)


class GetModelStatus(object):

    def __init__(self, models):
        self.models = models

    def on_get(self, req, resp, model_name, requested_version=0):
        logger.debug("MODEL_STATUS, get request: {}, {}"
                     .format(model_name, requested_version))
        valid_model_status = check_availability_of_requested_status(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_status:
            resp.status = falcon.HTTP_NOT_FOUND
            logger.debug("MODEL_STATUS, invalid model spec from request")
            err_out_json = {
                'error': WRONG_MODEL_SPEC.format(model_name,
                                                 requested_version)
            }
            resp.body = json.dumps(err_out_json)
            return
        requested_version = int(requested_version)

        response = get_model_status_pb2.GetModelStatusResponse()
        if requested_version:
            version_status = self.models[model_name].versions_statuses[
                requested_version]
            add_status_to_response(version_status, response)
        else:
            for version_status in self.models[model_name].versions_statuses. \
                    values():
                add_status_to_response(version_status, response)
        logger.debug("MODEL_STATUS created a response for {} - {}"
                     .format(model_name, requested_version))
        resp.status = falcon.HTTP_200
        resp.body = MessageToJson(response,
                                  including_default_value_fields=True)


class GetModelMetadata(object):

    def __init__(self, models):
        self.models = models

    def on_get(self, req, resp, model_name, requested_version=0):
        logger.debug("MODEL_METADATA, get request: {}, {}"
                     .format(model_name, requested_version))
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            resp.status = falcon.HTTP_NOT_FOUND
            logger.debug("MODEL_METADATA, invalid model spec from request")
            err_out_json = {
                'error': WRONG_MODEL_SPEC.format(model_name,
                                                 requested_version)
            }
            resp.body = json.dumps(err_out_json)
            return

        target_engine = self.models[model_name].engines[version]

        inputs = target_engine.net.inputs
        outputs = target_engine.net.outputs

        signature_def = prepare_get_metadata_output(inputs=inputs,
                                                    outputs=outputs,
                                                    model_keys=target_engine.
                                                    model_keys)
        response = get_model_metadata_pb2.GetModelMetadataResponse()

        model_data_map = get_model_metadata_pb2.SignatureDefMap()
        model_data_map.signature_def['serving_default'].CopyFrom(
            signature_def)
        response.metadata['signature_def'].Pack(model_data_map)
        response.model_spec.name = model_name
        response.model_spec.version.value = version
        logger.debug("MODEL_METADATA created a response for {} - {}"
                     .format(model_name, version))
        resp.status = falcon.HTTP_200
        resp.body = MessageToJson(response)


class Predict():

    def __init__(self, models):
        self.models = models

    def on_post(self, req, resp, model_name, requested_version=0):
        valid_model_spec, version = check_availability_of_requested_model(
            models=self.models, requested_version=requested_version,
            model_name=model_name)

        if not valid_model_spec:
            resp.status = falcon.HTTP_NOT_FOUND
            logger.debug("PREDICT, invalid model spec from request, "
                         "{} - {}".format(model_name, requested_version))
            err_out_json = {
                'error': WRONG_MODEL_SPEC.format(model_name,
                                                 requested_version)
            }
            resp.body = json.dumps(err_out_json)
            return
        body = req.media
        if type(body) is not dict:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({'error': 'Invalid JSON in request body'})
            return

        target_engine = self.models[model_name].engines[version]
        input_format = get_input_format(body, target_engine.input_key_names)
        if input_format == INVALID_FORMAT:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({'error': 'Invalid inputs in request '
                                             'body'})
            return

        inputs = preprocess_json_request(body, input_format,
                                         target_engine.input_key_names)

        deserialization_start_time = datetime.datetime.now()
        inference_input, error_message = \
            prepare_input_data(target_engine=target_engine, data=inputs,
                               service_type=REST)
        duration = (datetime.datetime.now() -
                    deserialization_start_time).total_seconds() * 1000
        logger.debug("PREDICT; input deserialization completed; {}; {}; {} ms"
                     .format(model_name, version, duration))
        if error_message is not None:
            resp.status = code = statusCodes['invalid_arg'][REST]
            err_out_json = {'error': error_message}
            logger.debug("PREDICT, problem with input data. Exit code {}"
                         .format(code))
            resp.body = json.dumps(err_out_json)
            return
        inference_request = Request(inference_input)
        target_engine.requests_queue.put(inference_request)
        inference_output, used_ireq_index = inference_request.wait_for_result()
        if type(inference_output) is str:
            resp.status = falcon.HTTP_400
            err_out_json = {'error': inference_output}
            resp.body = json.dumps(err_out_json)
            target_engine.free_ireq_index_queue.put(used_ireq_index)
            return
        serialization_start_time = datetime.datetime.now()
        for key, value in inference_output.items():
            inference_output[key] = value.tolist()

        response = prepare_json_response(
            OUTPUT_REPRESENTATION[input_format], inference_output,
            target_engine.model_keys['outputs'])
        resp.status = falcon.HTTP_200
        resp.body = json.dumps(response)
        duration = (datetime.datetime.now() -
                    serialization_start_time).total_seconds() * 1000
        logger.debug("PREDICT; inference results serialization completed;"
                     " {}; {}; {} ms".format(model_name, version, duration))
        target_engine.free_ireq_index_queue.put(used_ireq_index)
        return


def create_rest_api(models):
    app = falcon.API()
    get_model_status = GetModelStatus(models)
    get_model_meta = GetModelMetadata(models)
    predict = Predict(models)

    app.add_route('/v1/models/{model_name}', get_model_status)
    app.add_route('/v1/models/{model_name}/'
                  'versions/{requested_version}',
                  get_model_status)

    app.add_route('/v1/models/{model_name}/metadata', get_model_meta)
    app.add_route('/v1/models/{model_name}/'
                  'versions/{requested_version}/metadata',
                  get_model_meta)

    app.add_route('/v1/models/{model_name}:predict', predict)
    app.add_route('/v1/models/{model_name}/versions/'
                  '{requested_version}:predict',
                  predict)
    return app
