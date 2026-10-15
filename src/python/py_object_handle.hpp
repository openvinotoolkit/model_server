//*****************************************************************************
// Copyright 2026 Intel Corporation
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

namespace ovms {

// MediaPipe packet payload owning a Python object reference created by libpython_calculators.
class PyObjectHandle {
public:
    using ReleaseFn = void (*)(void*);

    PyObjectHandle(void* object, ReleaseFn release) :
        object(object),
        release(release) {}
    PyObjectHandle(const PyObjectHandle&) = delete;
    PyObjectHandle& operator=(const PyObjectHandle&) = delete;
    ~PyObjectHandle() {
        if (object != nullptr) {
            release(object);
        }
    }

    void* get() const { return object; }

private:
    void* object;
    ReleaseFn release;
};

}  // namespace ovms
