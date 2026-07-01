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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Lists models available on the OpenVINO Model Server via '
                    'the OpenAI-compatible GET /v3/models endpoint.')
    parser.add_argument('--http_address', default='localhost',
                        help='Specify url to HTTP service. default: localhost')
    parser.add_argument('--http_port', default=8000, type=int,
                        help='Specify port to HTTP service. default: 8000')
    parser.add_argument('--timeout', default=10, type=int,
                        help='HTTP request timeout in seconds. default: 10')

    args = vars(parser.parse_args())

    url = "http://{}:{}/v3/models".format(args['http_address'], args['http_port'])

    try:
        with urllib.request.urlopen(url, timeout=args['timeout']) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"Request failed: HTTP {exc.code} {exc.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Could not reach server at {url}: {exc}", file=sys.stderr)
        sys.exit(1)

    models = body.get("data", [])
    if not models:
        print("No models currently served.")
    else:
        print(f"{'MODEL ID':<50}  {'OWNED BY'}")
        print("-" * 62)
        for m in models:
            print(f"{m.get('id', ''):<50}  {m.get('owned_by', '')}")
