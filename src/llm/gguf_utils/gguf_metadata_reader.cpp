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
#include "gguf_metadata_reader.hpp"

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <stdexcept>

namespace ovms {
namespace {
constexpr uint64_t MAX_METADATA_SIZE_BYTES = 64 * 1024 * 1024;
constexpr uint64_t MAX_STRING_SIZE_BYTES = 1024 * 1024;
constexpr uint64_t BITS_PER_BYTE = 8;
constexpr uint64_t UINT32_SIZE_BYTES = 4;
constexpr uint64_t UINT64_SIZE_BYTES = 8;
constexpr uint64_t MIN_METADATA_ENTRY_SIZE_BYTES = UINT64_SIZE_BYTES + UINT32_SIZE_BYTES;

constexpr uint64_t GGUF_MAGIC = 0x46554747;
constexpr uint64_t GGUF_VERSION_2 = 2;
constexpr uint64_t GGUF_VERSION_3 = 3;

constexpr uint64_t GGUF_TYPE_UINT8 = 0;
constexpr uint64_t GGUF_TYPE_INT8 = 1;
constexpr uint64_t GGUF_TYPE_UINT16 = 2;
constexpr uint64_t GGUF_TYPE_INT16 = 3;
constexpr uint64_t GGUF_TYPE_UINT32 = 4;
constexpr uint64_t GGUF_TYPE_INT32 = 5;
constexpr uint64_t GGUF_TYPE_FLOAT32 = 6;
constexpr uint64_t GGUF_TYPE_BOOL = 7;
constexpr uint64_t GGUF_TYPE_STRING = 8;
constexpr uint64_t GGUF_TYPE_ARRAY = 9;
constexpr uint64_t GGUF_TYPE_UINT64 = 10;
constexpr uint64_t GGUF_TYPE_INT64 = 11;
constexpr uint64_t GGUF_TYPE_FLOAT64 = 12;
constexpr uint64_t GGUF_TYPE_COUNT = GGUF_TYPE_FLOAT64 + 1;

constexpr uint64_t UINT8_SIZE_BYTES = 1;
constexpr uint64_t UINT16_SIZE_BYTES = 2;
constexpr uint64_t UINT32_VALUE_SIZE_BYTES = 4;
constexpr uint64_t UINT64_VALUE_SIZE_BYTES = 8;

uint64_t getFixedValueSize(uint64_t type) {
    switch (type) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_BOOL:
        return UINT8_SIZE_BYTES;
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
        return UINT16_SIZE_BYTES;
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
        return UINT32_VALUE_SIZE_BYTES;
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
        return UINT64_VALUE_SIZE_BYTES;
    default:
        return 0;
    }
}
}  // namespace

GgufMetadataReader::GgufMetadataReader(const std::string& path) :
    file(path, std::ios::binary),
    remaining(std::min<uint64_t>(std::filesystem::file_size(path), MAX_METADATA_SIZE_BYTES)) {
    if (!file.is_open())
        throw std::runtime_error("Cannot open GGUF metadata");
}

void GgufMetadataReader::skip(uint64_t bytes) {
    if (bytes > remaining)
        throw std::runtime_error("Truncated or oversized GGUF metadata");
    file.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!file)
        throw std::runtime_error("Cannot seek GGUF metadata");
    remaining -= bytes;
}

uint64_t GgufMetadataReader::readInteger(std::size_t bytes) {
    if (bytes > remaining)
        throw std::runtime_error("Truncated or oversized GGUF metadata");
    uint64_t value = 0;
    for (std::size_t index = 0; index < bytes; ++index) {
        int byte = file.get();
        if (byte == std::char_traits<char>::eof())
            throw std::runtime_error("Truncated GGUF metadata");
        value |= static_cast<uint64_t>(byte) << (BITS_PER_BYTE * index);
    }
    remaining -= bytes;
    return value;
}

std::string GgufMetadataReader::readString() {
    uint64_t length = readInteger(UINT64_SIZE_BYTES);
    if (length > remaining || length > MAX_STRING_SIZE_BYTES)
        throw std::runtime_error("Truncated or oversized GGUF metadata string");
    std::string value(static_cast<std::size_t>(length), '\0');
    file.read(value.data(), static_cast<std::streamsize>(length));
    if (!file)
        throw std::runtime_error("Truncated GGUF metadata string");
    remaining -= length;
    return value;
}

void GgufMetadataReader::skipValue(uint64_t type) {
    if (type >= GGUF_TYPE_COUNT)
        throw std::runtime_error("Invalid GGUF metadata type");
    if (type == GGUF_TYPE_STRING) {
        skip(readInteger(UINT64_SIZE_BYTES));
    } else if (type == GGUF_TYPE_ARRAY) {
        uint64_t elementType = readInteger(UINT32_SIZE_BYTES);
        uint64_t count = readInteger(UINT64_SIZE_BYTES);
        if (elementType >= GGUF_TYPE_COUNT || elementType == GGUF_TYPE_ARRAY)
            throw std::runtime_error("Invalid GGUF metadata array type");
        if (elementType == GGUF_TYPE_STRING) {
            if (count > remaining / UINT64_SIZE_BYTES)
                throw std::runtime_error("Oversized GGUF metadata array");
            for (uint64_t index = 0; index < count; ++index)
                skip(readInteger(UINT64_SIZE_BYTES));
        } else {
            uint64_t elementSize = getFixedValueSize(elementType);
            if (count > remaining / elementSize)
                throw std::runtime_error("Oversized GGUF metadata array");
            skip(count * elementSize);
        }
    } else {
        skip(getFixedValueSize(type));
    }
}

std::string GgufMetadataReader::readArchitecture() {
    if (readInteger(UINT32_SIZE_BYTES) != GGUF_MAGIC)
        throw std::runtime_error("Invalid GGUF magic");
    uint64_t version = readInteger(UINT32_SIZE_BYTES);
    if (version != GGUF_VERSION_2 && version != GGUF_VERSION_3)
        throw std::runtime_error("Unsupported GGUF metadata version");
    uint64_t tensorCount = readInteger(UINT64_SIZE_BYTES);
    uint64_t metadataCount = readInteger(UINT64_SIZE_BYTES);
    if (tensorCount == 0 || metadataCount > remaining / MIN_METADATA_ENTRY_SIZE_BYTES)
        throw std::runtime_error("Invalid GGUF metadata header");
    std::string architecture;
    for (uint64_t index = 0; index < metadataCount; ++index) {
        std::string key = readString();
        uint64_t type = readInteger(UINT32_SIZE_BYTES);
        if (key == "general.architecture") {
            if (type != GGUF_TYPE_STRING || !architecture.empty())
                throw std::runtime_error("Invalid GGUF architecture metadata");
            architecture = readString();
        } else {
            skipValue(type);
        }
    }
    if (architecture.empty())
        throw std::runtime_error("Missing GGUF architecture metadata");
    return architecture;
}
}  // namespace ovms
