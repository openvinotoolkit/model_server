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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <variant>

#include "src/port/rapidjson_stringbuffer.hpp"
#include "src/port/rapidjson_writer.hpp"

namespace ovms {

// Key-value metadata of a servable mirroring JSON: bool, integer, double, string and nested object values.
// Filled by node initializers while a graph is being initialized, then published as an immutable snapshot.
class GraphMetadata {
public:
    using Value = std::variant<bool, int64_t, double, std::string, std::shared_ptr<const GraphMetadata>>;

    // Each setter returns false and keeps the existing value if the key is already set.
    bool setBool(const std::string& key, bool value);
    bool setInt(const std::string& key, int64_t value);
    bool setDouble(const std::string& key, double value);
    bool setString(const std::string& key, const std::string& value);
    bool setObject(const std::string& key, std::shared_ptr<const GraphMetadata> value);

    const Value* find(const std::string& key) const;
    bool empty() const;

    // Writes members only, so they can be added to an already started JSON object. Keys in skipKeys are left out.
    void writeMembers(rapidjson::Writer<rapidjson::StringBuffer>& writer, const std::unordered_set<std::string>& skipKeys) const;
    // Writes the metadata as a complete JSON object.
    void writeObject(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

private:
    bool set(const std::string& key, Value value);

    std::map<std::string, Value> values;
};

}  // namespace ovms
