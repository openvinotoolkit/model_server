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


def test_get_model_status_successful(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/test',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 200


def test_get_model_status_successful_with_specific_version(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/test/versions/2',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 200


def test_get_model_status_wrong_model(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/fake_model',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 404


def test_get_model_status_wrong_version(client):
    response = client.simulate_request(method='GET',
                                       path='/v1/models/test/versions/5',
                                       headers={
                                           "Content-Type":
                                               "application/json"})
    assert response.status_code == 404
