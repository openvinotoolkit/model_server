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

import falcon

from src.api.models.model_builder import ModelBuilder


def create_dispatcher(available_models: list, ovms_port: int):
    dispatch_map = {}
    for available_model in available_models:
        model = ModelBuilder.build_model(available_model, ovms_port)
        dispatch_map[model.endpoint] = model

    dispatcher = falcon.API()

    for target_model, request_handler in dispatch_map.items():
        dispatcher.add_route(f"/{target_model}", request_handler)

    return dispatcher
