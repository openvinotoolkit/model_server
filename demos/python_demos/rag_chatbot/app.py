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

import gradio as gr
import argparse

from streamer import OvmsStreamer


parser = argparse.ArgumentParser(description='Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot')

parser.add_argument('--web_url',
                    required=True,
                    help='Web server URL')
parser.add_argument('--ovms_url',
                    required=True,
                    help='OVMS server URL')
args = parser.parse_args()


def callback(message, history):
    streamer = OvmsStreamer(args.ovms_url.split(':')[0], int(args.ovms_url.split(':')[1]))
    streamer.request_async(message)
    result = ""
    for completion in streamer:
        print(completion, end='', flush=True)
        result += completion
        yield result

gr.ChatInterface(callback).queue(concurrency_count=16).launch(server_name=args.web_url.split(':')[0], server_port=int(args.web_url.split(':')[1]))
