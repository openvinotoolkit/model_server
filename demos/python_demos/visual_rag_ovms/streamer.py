#*****************************************************************************
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

import tritonclient.grpc as grpcclient
from tritonclient.utils import serialize_byte_tensor
import numpy as np
import queue


def serialize_prompts(prompts):
    infer_input = grpcclient.InferInput("pre_prompt", [len(prompts)], "BYTES")
    if len(prompts) == 1:
        # Single batch serialized directly as bytes
        infer_input._raw_content = prompts[0].encode()
        return infer_input
    # Multi batch serialized in tritonclient 4byte len format
    infer_input._raw_content = serialize_byte_tensor(
        np.array(prompts, dtype=np.object_)).item()
    return infer_input


channel_args = [
    # Do not drop the connection for long workloads
    ("grpc.http2.max_pings_without_data", 0),
]


class OvmsStreamer:
    def __init__(self, server_name, server_port):
        self.server_name = server_name
        self.server_port = server_port
        self.client = grpcclient.InferenceServerClient(f"{server_name}:{server_port}", channel_args=channel_args, verbose=False)
        self.result_queue = queue.Queue()

    def request_async(self, question):
        def callback(result, error):
            if error:
                # raise error
                self.result_queue.put(0)
            if result.as_numpy('end_signal') is not None:
                self.result_queue.put(0)
            elif result.as_numpy('completion') is not None:
                self.result_queue.put(result.as_numpy('completion').tobytes().decode())
            else:
                assert False, "unexpected output"
        self.client.start_stream(callback=callback)
        infer_input = serialize_prompts([question])
        self.client.async_stream_infer(model_name="python_model", inputs=[infer_input])


    def __iter__(self):
        return self

    def __next__(self):
        item = self.result_queue.get()
        if isinstance(item, int):
            self.client.stop_stream()
            raise StopIteration
        if not isinstance(item, str):
            assert False, "must be int or str"
        return item
