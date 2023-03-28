#
# Copyright (c) 2023 Intel Corporation
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

BLINGFIRE_SHA=5089d31914cbed7a24589e753bd6cd362a377fbb

.PHONY: setup

default: setup

setup:
# Build custom node
	cd ../../../src/custom_nodes/tokenizer && \
	make BASE_OS=ubuntu
	mkdir -p workspace/lib && \
	cp ../../../src/custom_nodes/tokenizer/lib/ubuntu/libdetokenizer.so workspace/lib/libdetokenizer.so && \
	cp ../../../src/custom_nodes/tokenizer/lib/ubuntu/libtokenizer.so workspace/lib/libtokenizer.so
# Prepare tokenization models
	mkdir -p workspace/tokenizers
	wget https://github.com/microsoft/BlingFire/raw/${BLINGFIRE_SHA}/ldbsrc/ldb/gpt2.bin -O workspace/tokenizers/gpt2.bin
	wget https://github.com/microsoft/BlingFire/raw/${BLINGFIRE_SHA}/ldbsrc/ldb/gpt2.i2w -O workspace/tokenizers/gpt2.i2w
# Copy configuration file to workspace directory
	cp config.json workspace/.

clean:
	@rm -rf workspace