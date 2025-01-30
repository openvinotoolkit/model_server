//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#pragma once

#include <memory>
#include <string>
#include <utility>

#include "nodesessionmetadata.hpp"

namespace ovms {
class NodeInputHandler;
struct NodeOutputHandler;
class Status;
class TensorWithSource;
template <unsigned int N>
class Timer;

enum : unsigned int {
    GET_INFER_REQUEST,
    EXECUTE,
    TIMER_END
};

class NodeSession {
    NodeSessionMetadata metadata;
    session_key_t sessionKey;
    const std::string& nodeName;

protected:
    std::unique_ptr<Timer<TIMER_END>> timer;
    std::unique_ptr<NodeInputHandler> inputHandler;

public:
    NodeSession(const NodeSessionMetadata& metadata, const std::string& nodeName, uint32_t inputsCount, const CollapseDetails& collapsingDetails);
    virtual ~NodeSession();
    const std::string& getName() const { return nodeName; }
    Status setInput(const std::string& inputName, TensorWithSource& tensor, session_id_t shardId);
    const NodeSessionMetadata& getNodeSessionMetadata() const;
    const session_key_t& getSessionKey() const { return sessionKey; }
    bool isReady() const;
    virtual void release() {}
    virtual bool tryDisarm(uint32_t microseconds) { return true; }
    Status notifyFinishedDependency();
    Timer<TIMER_END>& getTimer() const;
};

class ReleaseSessionGuard {
    NodeSession& nodeSession;

public:
    ReleaseSessionGuard(NodeSession& nodeSession);
    ~ReleaseSessionGuard();
};
}  // namespace ovms
