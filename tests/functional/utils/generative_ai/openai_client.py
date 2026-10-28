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

# pylint: disable=not-an-iterable

import json
import sys
from argparse import ArgumentParser

from openai import OpenAI


def run_inference(parameters):
    base_url = f"http://localhost:{parameters.port}/v3"
    client = OpenAI(
        base_url=base_url,
        api_key=base_url
    )

    model = parameters.model
    max_tokens = int(parameters.max_tokens)
    stream = bool(parameters.stream)
    temperature = int(parameters.temperature)
    image_path = parameters.image_path
    size = parameters.size
    n = int(parameters.n)
    num_inference_steps = int(parameters.num_inference_steps)
    rng_seed = int(parameters.rng_seed)
    strength = float(parameters.strength)
    timeout = int(parameters.timeout)

    with open(parameters.input_data_file, encoding="utf-8") as fo:
        input_data = json.load(fo)
    stream_content = []

    if parameters.endpoint == "completions":
        response = client.completions.create(
            model=model,
            prompt=input_data,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            timeout=timeout,
        )
        if stream:
            for r in response:
                for choice in r.choices:
                    if choice.text is not None:
                        stream_content.append(choice.text)
            outputs_content = ["".join(stream_content)]
        else:
            outputs_content = [response.choices[0].text]
    elif parameters.endpoint == "chat/completions":
        response = client.chat.completions.create(
            model=model,
            messages=input_data,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            timeout=timeout,
        )
        if stream:
            for r in response:
                for choice in r.choices:
                    if choice.delta.content is not None:
                        stream_content.append(choice.delta.content)
            outputs_content = ["".join(stream_content)]
        else:
            outputs_content = [response.choices[0].message.content]
    elif parameters.endpoint == "images/edits":
        response = client.images.edit(
            model=model,
            prompt=input_data,
            image=open(image_path, "rb"),   # pylint: disable=consider-using-with
            size=size,
            n=n,
            extra_body={
                "num_inference_steps": num_inference_steps,
                "rng_seed": rng_seed,
                "strength": strength,
            },
            timeout=timeout,
        )
        outputs_content = [response.data[0].b64_json]
    elif parameters.endpoint == "images/generations":
        response = client.images.generate(
            model=model,
            prompt=input_data,
            size=size,
            n=n,
            extra_body={
                "num_inference_steps": num_inference_steps,
                "rng_seed": rng_seed,
            },
            timeout=timeout,
        )
        outputs_content = [response.data[0].b64_json]
    else:
        raise NotImplementedError()

    print(outputs_content)


if __name__ == '__main__':
    parser = ArgumentParser(description="Python script for running inference using OpenAI client")
    parser.add_argument(
        "--port",
        help="port",
        required=True,
    )
    parser.add_argument(
        "--api_key",
        help="api key",
        default="unused",
        required=False,
    )
    parser.add_argument(
        "--model",
        help="Model name for inference",
        required=True,
    )
    parser.add_argument(
        "--endpoint",
        help="Completions or chat completions",
        required=True,
    )
    parser.add_argument(
        "--input_data_file",
        help="File with prompt/message in json form",
        required=True,
    )
    parser.add_argument(
        "--max_tokens",
        help="max_tokens",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--stream",
        help="stream",
        default=True,
        required=False,
    )
    parser.add_argument(
        "--temperature",
        help="temperature",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--image_path",
        help="image_path",
        required=False,
    )
    parser.add_argument(
        "--size",
        help="size",
        default="512x512",
        required=False,
    )
    parser.add_argument(
        "--n",
        help="n",
        default=1,
        required=False,
    )
    parser.add_argument(
        "--num_inference_steps",
        help="num_inference_steps",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--rng_seed",
        help="rng_seed",
        default=42,
        required=False,
    )
    parser.add_argument(
        "--strength",
        help="strength",
        default=0.7,
        required=False,
    )
    parser.add_argument(
        "--timeout",
        help="timeout",
        default=2400,
        required=False,
    )

    args = parser.parse_args()

    if len(sys.argv) < 2:
        args = parser.parse_args(["-h"])
    else:
        run_inference(args)
