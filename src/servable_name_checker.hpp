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

#include <string>
#include <type_traits>

namespace ovms {

enum class ServableQueryType : uint8_t {
    Model = 1 << 0,
    Pipeline = 1 << 1,
    Mediapipe = 1 << 2,
    All = Model | Pipeline | Mediapipe
};

inline ServableQueryType operator|(ServableQueryType a, ServableQueryType b) {
    using T = std::underlying_type_t<ServableQueryType>;
    return static_cast<ServableQueryType>(static_cast<T>(a) | static_cast<T>(b));
}

inline bool hasFlag(ServableQueryType value, ServableQueryType flag) {
    using T = std::underlying_type_t<ServableQueryType>;
    return (static_cast<T>(value) & static_cast<T>(flag)) != 0;
}

class ServableNameChecker {
public:
    virtual ~ServableNameChecker() = default;
    virtual bool servableExists(const std::string& name, ServableQueryType check = ServableQueryType::All) const = 0;
};

}  // namespace ovms
