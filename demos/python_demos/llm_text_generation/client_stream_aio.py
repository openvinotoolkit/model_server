#!/usr/bin/env python
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import asyncio
import queue
import sys
import uuid

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException

async def async_stream_yield(prompts):
    i = 0
    for prompt in prompts:
        print(prompts[i])
        inputs = []
        inputs.append(grpcclient.InferInput("pre_prompt", [1], "BYTES"))
        inputs[0]._raw_content = prompts[i].encode()
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("completion"))
        yield {
            "model_name": "python_model",
            "inputs": inputs,
            "outputs": outputs,
        }
        i = i + 1


async def main(FLAGS):
    prompts = FLAGS.prompt
    print("NUMBER OF PROMPTS {}".format(len(prompts)))
    channel_args = [
        # Do not drop the connection for long workloads
        ("grpc.http2.max_pings_without_data", 0),
    ]
    async with grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose,  channel_args=channel_args
    ) as triton_client:
        try:
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_stream_yield(prompts),
                stream_timeout=FLAGS.stream_timeout,
            )
            async for response in response_iterator:
                if response is not None and response[0].as_numpy('end_signal') is None:
                    print(response[0].as_numpy('completion').tobytes().decode(), flush=True, end='')
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    print("\n\nPASS: grpc aio sequence stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and it gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument('-p', '--prompt',
                    required=True,
                    default=[],
                    action="append",
                    help='Questions for the endpoint to answer')
    FLAGS = parser.parse_args()
    asyncio.run(main(FLAGS))