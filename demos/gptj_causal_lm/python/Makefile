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

BUILD_CUSTOM_NODE=false
BASE_OS=ubuntu

setup:
ifeq ($(BUILD_CUSTOM_NODE),true)
# Build custom node
	cd ../../../src/custom_nodes/tokenizer && \
	make BASE_OS=$(BASE_OS)
	mkdir -p workspace/lib && \
	cp ../../../src/custom_nodes/tokenizer/lib/$(BASE_OS)/libdetokenizer.so workspace/lib/libdetokenizer.so && \
	cp ../../../src/custom_nodes/tokenizer/lib/$(BASE_OS)/libtokenizer.so workspace/lib/libtokenizer.so
endif
# Prepare tokenization models
	mkdir -p workspace/tokenizers
	wget https://github.com/microsoft/BlingFire/raw/${BLINGFIRE_SHA}/ldbsrc/ldb/gpt2.bin -O workspace/tokenizers/gpt2.bin
	wget https://github.com/microsoft/BlingFire/raw/${BLINGFIRE_SHA}/ldbsrc/ldb/gpt2.i2w -O workspace/tokenizers/gpt2.i2w
# Copy configuration file to workspace directory
	cp config.json workspace/.
ifeq ($(BUILD_CUSTOM_NODE),false)
	sed -i 's:\/workspace\/lib\/libtokenizer.so:\/ovms\/lib\/custom_nodes\/libtokenizer.so:g' workspace/config.json
	sed -i 's:\/workspace\/lib\/libdetokenizer.so:\/ovms\/lib\/custom_nodes\/libdetokenizer.so:g' workspace/config.json
endif

clean:
	@rm -rf workspace