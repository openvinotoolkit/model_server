#!/usr/bin/env python

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