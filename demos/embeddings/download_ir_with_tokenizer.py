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
import os
import shutil
import subprocess
import sys

from huggingface_hub import list_repo_files, snapshot_download

OV_SUBDIR = "openvino"

parser = argparse.ArgumentParser(description='Download OpenVINO IR of an embedding model from HuggingFace and generate tokenizer/detokenizer')
parser.add_argument('--model_id', required=True, help='HuggingFace model id, e.g. thenlper/gte-small')
parser.add_argument('--output_dir', default='C:/models' if os.name == 'nt' else os.path.expanduser('~/models'),
                    help='Root directory where the model will be stored (default: C:/models on Windows, ${HOME}/models on Linux)')
parser.add_argument('--revision', default=None, help='Model revision (branch, tag or commit)')
parser.add_argument('--trust_remote_code', action='store_true', help='Pass --trust-remote-code to convert_tokenizer')
args = parser.parse_args()

repo_files = list_repo_files(args.model_id, revision=args.revision)
if not any(f.startswith(OV_SUBDIR + "/") for f in repo_files):
    print(f"Repository {args.model_id} does not contain '{OV_SUBDIR}' directory. Stopping.")
    sys.exit(1)

model_dir = os.path.join(args.output_dir, args.model_id)
download_dir = os.path.join(model_dir, "_download")
print(f"Downloading '{OV_SUBDIR}' directory of {args.model_id}")
snapshot_download(repo_id=args.model_id, revision=args.revision, allow_patterns=[f"{OV_SUBDIR}/*"], local_dir=download_dir)

# Flatten <model_dir>/_download/openvino/* into <model_dir>
src_dir = os.path.join(download_dir, OV_SUBDIR)
for name in os.listdir(src_dir):
    shutil.move(os.path.join(src_dir, name), os.path.join(model_dir, name))
shutil.rmtree(download_dir)

if shutil.which("convert_tokenizer") is None:
    print("convert_tokenizer not found. Install it with: pip install openvino-tokenizers[transformers]")
    sys.exit(1)

cmd = ["convert_tokenizer", args.model_id, "--with-detokenizer", "-o", model_dir]
if args.revision:
    cmd += ["--revision", args.revision]
if args.trust_remote_code:
    cmd.append("--trust-remote-code")
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

print(f"Model with tokenizer and detokenizer ready in {model_dir}")
