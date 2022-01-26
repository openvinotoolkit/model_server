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

import numpy as np

def print_statistics(processing_times, batch_size):
    print('\nprocessing time for all iterations')
    print('average time: {:.2f} ms; average speed: {:.2f} fps'
          .format(round(np.average(processing_times), 2),
                  round(1000 * batch_size / np.average(processing_times), 2)))

    print('median time: {:.2f} ms; median speed: {:.2f} fps'
          .format(round(np.median(processing_times), 2),
                 round(1000 * batch_size / np.median(processing_times), 2)))

    print('max time: {:.2f} ms; min speed: {:.2f} fps'.format(round(np.max(processing_times), 2),round(1000 * batch_size / np.max(processing_times), 2)))

    print('min time: {:.2f} ms; max speed: {:.2f} fps'.format(round(np.min(processing_times), 2),
                                                          round(1000 * batch_size / np.min(processing_times), 2)))

    print('time percentile 90: {:.2f} ms; speed percentile 90: {:.2f} fps'.format(
    round(np.percentile(processing_times, 90), 2),
    round(1000 * batch_size / np.percentile(processing_times, 90), 2)
))
    print('time percentile 50: {:.2f} ms; speed percentile 50: {:.2f} fps'.format(
    round(np.percentile(processing_times, 50), 2),
    round(1000 * batch_size / np.percentile(processing_times, 50), 2)))
    print('time standard deviation: {:.2f}'.format(round(np.std(processing_times), 2)))
    print('time variance: {:.2f}'.format(round(np.var(processing_times), 2)))

def prepare_certs(server_cert=None, client_key=None, client_ca=None):
    if server_cert is not None:
        with open(server_cert, 'rb') as f:
            server_cert = f.read()
    if client_key is not None:
        with open(client_key, 'rb') as key:
            client_key = key.read()
    if client_ca is not None:
        with open(client_ca, 'rb') as ca:
            client_ca = ca.read()
    return server_cert, client_key, client_ca
