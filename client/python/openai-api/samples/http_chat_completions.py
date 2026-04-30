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

import argparse
import json
import sys
import urllib.error
import urllib.request


def wait_server_ready(base_url: str, timeout: int) -> None:
    url = f"{base_url}/v2/health/ready"
    try:
        urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        print(f"Server not ready: HTTP {exc.code}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Could not reach server at {url}: {exc}", file=sys.stderr)
        sys.exit(1)


def chat_completions(base_url: str, model: str, message: str,
                     system: str, max_tokens: int, temperature: float,
                     timeout: int) -> None:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v3/chat/completions",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"Request failed: HTTP {exc.code} {exc.read().decode()}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(body, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sends a request to the OpenVINO Model Server '
                    'OpenAI-compatible chat/completions endpoint.')
    parser.add_argument('--http_address', default='localhost',
                        help='Specify url to HTTP service. default: localhost')
    parser.add_argument('--http_port', default=8000, type=int,
                        help='Specify port to HTTP service. default: 8000')
    parser.add_argument('--model_name', required=True,
                        help='Model name as configured in OVMS.')
    parser.add_argument('--message', default='What is the capital of France?',
                        help='User message to send. default: "What is the capital of France?"')
    parser.add_argument('--system_prompt', default='',
                        help='Optional system prompt.')
    parser.add_argument('--max_tokens', default=200, type=int,
                        help='Maximum tokens to generate. default: 200')
    parser.add_argument('--temperature', default=0.0, type=float,
                        help='Sampling temperature. default: 0.0')
    parser.add_argument('--timeout', default=60, type=int,
                        help='HTTP request timeout in seconds. default: 60')
    parser.add_argument('--no_health_check', action='store_true',
                        help='Skip the /v2/health/ready check before sending the request.')

    args = vars(parser.parse_args())

    base_url = "http://{}:{}".format(args['http_address'], args['http_port'])

    if not args['no_health_check']:
        wait_server_ready(base_url, args['timeout'])

    chat_completions(
        base_url=base_url,
        model=args['model_name'],
        message=args['message'],
        system=args['system_prompt'],
        max_tokens=args['max_tokens'],
        temperature=args['temperature'],
        timeout=args['timeout'],
    )
