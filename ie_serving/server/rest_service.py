import datetime
import falcon
import json

from ie_serving.server.rest_msg_processing import preprocess_json_request, \
    prepare_json_response
from ie_serving.server.rest_msg_validation import get_input_format
from ie_serving.tensorflow_serving_api import get_model_metadata_pb2
from google.protobuf.json_format import MessageToJson

from ie_serving.logger import get_logger
from ie_serving.server.service_utils import \
    check_availability_of_requested_model
from ie_serving.server.get_model_metadata_utils import \
    prepare_get_metadata_output
from ie_serving.server.constants import WRONG_MODEL_METADATA, INVALID_FORMAT, \
    OUTPUT_REPRESENTATION
from ie_serving.server.predict_utils import prepare_input_data

logger = get_logger(__name__)


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
                'error': WRONG_MODEL_METADATA.format(model_name,
                                                     requested_version)
            }
            resp.body = json.dumps(err_out_json)
            return
        self.models[model_name].engines[version].in_use.acquire()

        inputs = self.models[model_name].engines[version].input_tensors
        outputs = self.models[model_name].engines[version].output_tensors

        signature_def = prepare_get_metadata_output(inputs=inputs,
                                                    outputs=outputs,
                                                    model_keys=self.models
                                                    [model_name].
                                                    engines[version].
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
        self.models[model_name].engines[version].in_use.release()
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
                'error': WRONG_MODEL_METADATA.format(model_name,
                                                     requested_version)
            }
            resp.body = json.dumps(err_out_json)
            return
        body = req.media
        if type(body) is not dict:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({'error': 'Provided data is not JSON'})
            return
        start_time = datetime.datetime.now()
        input_format = get_input_format(body, self.models[
            model_name].engines[version].input_key_names)
        if input_format == INVALID_FORMAT:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({'error': 'Not valid inputs in request '
                                             'body'})
            return
        format_check_end_time = datetime.datetime.now()
        duration = \
            (format_check_end_time - start_time).total_seconds() * 1000
        logger.debug("PREDICT;[REST] input format check completed; "
                     "{}; {}; {}ms".format(model_name, version, duration))

        start_time = datetime.datetime.now()
        inputs = preprocess_json_request(body, input_format, self.models[
            model_name].engines[version].input_key_names)
        input_preprocess_end_time = datetime.datetime.now()
        duration = \
            (input_preprocess_end_time - start_time).total_seconds() * 1000
        logger.debug("PREDICT;[REST] input preprocess completed; {}; {}; {}ms"
                     .format(model_name, version, duration))

        self.models[model_name].engines[version].in_use.acquire()
        start_time = datetime.datetime.now()
        occurred_problem, inference_input, batch_size, code = \
            prepare_input_data(models=self.models, model_name=model_name,
                               version=version, data=inputs, rest=True)
        deserialization_end_time = datetime.datetime.now()
        duration = \
            (deserialization_end_time - start_time).total_seconds() * 1000
        logger.debug("PREDICT; input deserialization completed; {}; {}; {}ms"
                     .format(model_name, version, duration))
        if occurred_problem:
            resp.status = code
            err_out_json = {'error': inference_input}
            logger.debug("PREDICT, problem with input data. Exit code {}"
                         .format(code))
            self.models[model_name].engines[version].in_use.release()
            resp.body = json.dumps(err_out_json)
            return

        inference_start_time = datetime.datetime.now()
        try:
            inference_output = self.models[model_name].engines[version] \
                .infer(inference_input, batch_size)
        except ValueError as error:
            resp.status = falcon.HTTP_400
            err_out_json = {'error': 'Corrupted input data'}
            logger.debug("PREDICT, problem with inference. "
                         "Corrupted input: {}".format(error))
            self.models[model_name].engines[version].in_use.release()
            resp.body = json.dumps(err_out_json)
            return
        inference_end_time = datetime.datetime.now()
        duration = \
            (inference_end_time - inference_start_time).total_seconds() * 1000
        logger.debug("PREDICT; inference execution completed; {}; {}; {}ms"
                     .format(model_name, version, duration))
        for key, value in inference_output.items():
            inference_output[key] = value.tolist()

        response = prepare_json_response(
            OUTPUT_REPRESENTATION[input_format], inference_output,
            self.models[model_name].engines[version].model_keys['outputs'])

        resp.status = falcon.HTTP_200
        resp.body = json.dumps(response)
        serialization_end_time = datetime.datetime.now()
        duration = \
            (serialization_end_time -
             inference_end_time).total_seconds() * 1000
        logger.debug("PREDICT; inference results serialization completed;"
                     " {}; {}; {}ms".format(model_name, version, duration))
        self.models[model_name].engines[version].in_use.release()
        return


def create_rest_api(models):
    app = falcon.API()
    get_model_meta = GetModelMetadata(models)
    predict = Predict(models)
    app.add_route('/v1/models/{model_name}/metadata', get_model_meta)
    app.add_route('/v1/models/{model_name}/'
                  'versions/{requested_version}/metadata',
                  get_model_meta)
    app.add_route('/v1/models/{model_name}:predict', predict)
    app.add_route('/v1/models/{model_name}/versions/'
                  '{requested_version}:predict',
                  predict)
    return app
