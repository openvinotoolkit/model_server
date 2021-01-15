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
    const uint32_t sequenceControlInput;
    const uint64_t sequenceId;
    SequenceProcessingSpec(uint32_t sequenceControlInput, uint64_t sequenceId) :
        sequenceControlInput(sequenceControlInput),
        sequenceId(sequenceId) {}
};

// For now stateless ModelInstance does not need this class and it's not necessary for common execution path
// For stateful models SequenceProcessingSpec shall be sufficient
// I recommend removing this class after we get rid of dependencies that it created 
class ProcessingSpec {
private:
    SequenceProcessingSpec* sequenceProcessingSpec;

public:
    SequenceProcessingSpec* getSequenceProcessingSpecPtr() { return sequenceProcessingSpec; }

    void setSequenceProcessingSpec(uint32_t sequenceControlInput, uint64_t sequenceId) {
        sequenceProcessingSpec = new SequenceProcessingSpec(sequenceControlInput, sequenceId);
    }
};
}  // namespace ovms
