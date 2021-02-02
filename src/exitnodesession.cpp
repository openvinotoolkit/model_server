//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "exitnodesession.hpp"

#include <utility>

#include "logging.hpp"
#include "nodeinputhandler.hpp"

namespace ovms {
ExitNodeSession::ExitNodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, session_id_t shardsCount) :
    NodeSession(metadata, nodeName, inputsCount, shardsCount) {}

ExitNodeSession::ExitNodeSession(const NodeSessionMetadata&& metadata, const std::string& nodeName, uint32_t inputsCount, session_id_t shardsCount) :
    NodeSession(std::move(metadata), nodeName, inputsCount, shardsCount) {}

ExitNodeSession::~ExitNodeSession() = default;

void ExitNodeSession::release() {}

const BlobMap& ExitNodeSession::getInputBlobs() const {
    return this->inputHandler->getInputs();
}
}  // namespace ovms
