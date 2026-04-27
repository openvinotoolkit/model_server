#
# Copyright (c) 2026 Intel Corporation
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

import types

from tests.functional.utils.inference.capi import CapiServingWrapper
from tests.functional.utils.inference.communication.grpc import GRPC, GrpcCommunicationInterface
from tests.functional.utils.inference.communication.rest import REST, RestCommunicationInterface
from tests.functional.utils.inference.serving.cohere import COHERE, CohereWrapper
from tests.functional.utils.inference.serving.kf import KFS, KserveWrapper
from tests.functional.utils.inference.serving.openai import OPENAI, OpenAIWrapper
from tests.functional.utils.inference.serving.tf import TFS, TensorFlowServingWrapper
from tests.functional.utils.inference.serving.triton import TRITON, TritonServingWrapper
from tests.functional.constants.ovms_type import OvmsType
from ovms.object_model.ovsa import OvsaCerts


class InferenceClientFactory:
    @staticmethod
    def get_client(serving, communication, ovms_type=None):
        serving_class, communication_class = [None] * 2
        if ovms_type == OvmsType.CAPI:
            communication_class = CapiServingWrapper
        else:
            if serving == TFS:
                serving_class = TensorFlowServingWrapper
            elif serving == KFS:
                serving_class = KserveWrapper
            elif serving == TRITON:
                serving_class = TritonServingWrapper
            elif serving == OPENAI:
                serving_class = OpenAIWrapper
            elif serving == COHERE:
                serving_class = CohereWrapper
            else:
                raise Exception

            if serving in [TFS, KFS, TRITON, OPENAI, COHERE]:
                if communication == REST:
                    communication_class = RestCommunicationInterface
                elif communication == GRPC:
                    communication_class = GrpcCommunicationInterface
                else:
                    raise Exception

        # pylint: disable=too-many-arguments
        def common_inference_client_init(self,
                                         model=None,
                                         model_name=None,
                                         model_version=None,
                                         inputs: dict = None,
                                         input_names: list = None,
                                         input_data_types: dict = None,
                                         outputs: dict = None,
                                         output_names: list = None,
                                         output_data_types: dict = None,
                                         model_meta_from_serving: bool = None,
                                         is_mediapipe: bool = None,
                                         ssl_certificates: object = OvsaCerts.default_certs,
                                         context=None,
                                         **kwargs):
            """
                Aggregated __init__ method that will properly initialize any pair of communication & serving classes.
            """

            self.context = context

            # Copy simple parameters
            if model:
                self.model = model
                if model_meta_from_serving is None:
                    # If we got properly filled model description, do not fetch it via get_model_meta
                    model_meta_from_serving = False
            else:
                # use generic object as storage for parameters (version, name, inputs, outputs, etc.)
                # this step is required for @property methods.
                self.model = types.new_class("model_placeholder")()

                self.model.name = model_name
                self.model.version = model_version
                self.model.inputs = inputs if inputs else {}
                self.model.outputs = outputs if outputs else {}
                self.model.input_names = input_names if input_names else []
                self.model.output_names = output_names if output_names else []
                self.model.input_data_types = input_data_types if input_data_types else {}
                self.model.output_data_types = output_data_types if output_data_types else {}
                self.model.input_dims = {}
                self.model.output_dims = {}
                if is_mediapipe or (getattr(self.model, "name", None) is not None and "mediapipe" in self.model.name):
                    self.model.is_mediapipe = True
                else:
                    self.model.is_mediapipe = False

            # Init each parent class separate
            if serving_class:
                serving_class.__init__(self, model_meta_from_serving=model_meta_from_serving, **kwargs)
            if communication_class:
                if communication_class == CapiServingWrapper and kwargs.get("ovms_capi_instance", None) is None:
                    # Little trick to easily pass OvmsContext
                    kwargs["ovms_capi_instance"] = kwargs["port"]
                    del kwargs["port"]  # `port` is applicable only for GRPC/REST communication
                communication_class.__init__(self, ssl_certificates=ssl_certificates, **kwargs)
            else:
                self.type = communication

            # Initialize inference engine:
            self.create_inference()
        if ovms_type:
            name = f"{ovms_type.title()}InferenceClient"
            bases = (communication_class,)
        else:
            name = f"{serving.title()}{communication.title()}InferenceClient"
            bases = (serving_class,)
            if communication_class:
                bases += (communication_class,)

        inference_class_type = type(
            name,
            bases,
            {
                "__init__": common_inference_client_init,
            }
        )
        inference_class_type.serving = serving
        inference_class_type.communication = communication if communication else ovms_type
        return inference_class_type
