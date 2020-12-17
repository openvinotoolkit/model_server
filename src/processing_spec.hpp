//*****************************************************************************
// Copyright 2020 Intel Corporation
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

struct SequenceProcessingSpec {
    uint32_t sequenceControlInput;
    uint64_t sequenceId;
    SequenceProcessingSpec(uint32_t sequenceControlInput, uint64_t sequenceId) :
        sequenceControlInput(sequenceControlInput),
        sequenceId(sequenceId) {}
};

class ProcessingSpec {
private:
    std::shared_ptr<SequenceProcessingSpec> sequenceProcessingSpec;

public:
    std::shared_ptr<SequenceProcessingSpec> getSequenceProcessingSpecPtr() { return sequenceProcessingSpec; }

    void setSequenceProcessingSpec(uint32_t sequenceControlInput, uint64_t sequenceId) {
        sequenceProcessingSpec = std::make_shared<SequenceProcessingSpec>(sequenceControlInput, sequenceId);
    }
};
}  // namespace ovms
