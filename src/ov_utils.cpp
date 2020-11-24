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
#include "ov_utils.hpp"

#include <memory>

namespace ovms {

InferenceEngine::Blob::Ptr blobClone(const InferenceEngine::Blob::Ptr sourceBlob) {
    auto copyBlob = InferenceEngine::make_shared_blob<float>(sourceBlob->getTensorDesc());
    copyBlob->allocate();
    if (copyBlob->byteSize() != sourceBlob->byteSize()) {
        return nullptr;
    }
    std::memcpy((void*)copyBlob->buffer(), (void*)sourceBlob->buffer(), sourceBlob->byteSize());
    return copyBlob;
}

}  // namespace ovms
