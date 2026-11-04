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
#include "graph_metadata.hpp"

#include <utility>

namespace ovms {

bool GraphMetadata::set(const std::string& key, Value value) {
    return values.emplace(key, std::move(value)).second;
}

bool GraphMetadata::setBool(const std::string& key, bool value) {
    return set(key, Value(std::in_place_type<bool>, value));
}

bool GraphMetadata::setInt(const std::string& key, int64_t value) {
    return set(key, Value(std::in_place_type<int64_t>, value));
}

bool GraphMetadata::setDouble(const std::string& key, double value) {
    return set(key, Value(std::in_place_type<double>, value));
}

bool GraphMetadata::setString(const std::string& key, const std::string& value) {
    return set(key, Value(std::in_place_type<std::string>, value));
}

bool GraphMetadata::setObject(const std::string& key, std::shared_ptr<const GraphMetadata> value) {
    return set(key, Value(std::in_place_type<std::shared_ptr<const GraphMetadata>>, std::move(value)));
}

const GraphMetadata::Value* GraphMetadata::find(const std::string& key) const {
    auto it = values.find(key);
    return it == values.end() ? nullptr : &it->second;
}

bool GraphMetadata::empty() const {
    return values.empty();
}

void GraphMetadata::writeMembers(rapidjson::Writer<rapidjson::StringBuffer>& writer, const std::unordered_set<std::string>& skipKeys) const {
    for (const auto& [key, value] : values) {
        if (skipKeys.count(key)) {
            continue;
        }
        writer.String(key.c_str());
        if (const auto* boolValue = std::get_if<bool>(&value)) {
            writer.Bool(*boolValue);
        } else if (const auto* intValue = std::get_if<int64_t>(&value)) {
            writer.Int64(*intValue);
        } else if (const auto* doubleValue = std::get_if<double>(&value)) {
            writer.Double(*doubleValue);
        } else if (const auto* stringValue = std::get_if<std::string>(&value)) {
            writer.String(stringValue->c_str(), static_cast<rapidjson::SizeType>(stringValue->size()));
        } else {
            const auto& object = std::get<std::shared_ptr<const GraphMetadata>>(value);
            if (object) {
                object->writeObject(writer);
            } else {
                writer.Null();
            }
        }
    }
}

void GraphMetadata::writeObject(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    writeMembers(writer, {});
    writer.EndObject();
}

}  // namespace ovms
