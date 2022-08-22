//*****************************************************************************
// Copyright 2020-2021 Intel Corporation
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

namespace ovms {

class SequenceProcessingSpec {
    uint32_t sequenceControlInput = 0;
    uint64_t sequenceId = 0;

public:
    SequenceProcessingSpec() = default;
    SequenceProcessingSpec(uint32_t sequenceControlInput, uint64_t sequenceId) :
        sequenceControlInput(sequenceControlInput),
        sequenceId(sequenceId) {}
    const uint32_t getSequenceControlInput() const { return sequenceControlInput; }
    void setSequenceControlInput(uint32_t sequenceControlInput) { this->sequenceControlInput = sequenceControlInput; }
    const uint64_t getSequenceId() const { return sequenceId; }
    void setSequenceId(uint64_t sequenceId) { this->sequenceId = sequenceId; }
};
}  // namespace ovms
