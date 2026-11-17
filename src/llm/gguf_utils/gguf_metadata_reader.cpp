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
    remaining(std::min<uint64_t>(std::filesystem::file_size(path), 64 * 1024 * 1024)) {
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
        value |= static_cast<uint64_t>(byte) << (8 * index);
    }
    remaining -= bytes;
    return value;
}

std::string GgufMetadataReader::readString() {
    uint64_t length = readInteger(8);
    if (length > remaining || length > 1024 * 1024)
        throw std::runtime_error("Truncated or oversized GGUF metadata string");
    std::string value(static_cast<std::size_t>(length), '\0');
    file.read(value.data(), static_cast<std::streamsize>(length));
    if (!file)
        throw std::runtime_error("Truncated GGUF metadata string");
    remaining -= length;
    return value;
}

void GgufMetadataReader::skipValue(uint64_t type) {
    static constexpr uint64_t widths[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type >= std::size(widths))
        throw std::runtime_error("Invalid GGUF metadata type");
    if (type == 8) {
        skip(readInteger(8));
    } else if (type == 9) {
        uint64_t elementType = readInteger(4);
        uint64_t count = readInteger(8);
        if (elementType >= std::size(widths) || elementType == 9)
            throw std::runtime_error("Invalid GGUF metadata array type");
        if (elementType == 8) {
            if (count > remaining / 8)
                throw std::runtime_error("Oversized GGUF metadata array");
            for (uint64_t index = 0; index < count; ++index)
                skip(readInteger(8));
        } else {
            if (count > remaining / widths[elementType])
                throw std::runtime_error("Oversized GGUF metadata array");
            skip(count * widths[elementType]);
        }
    } else {
        skip(widths[type]);
    }
}

std::string GgufMetadataReader::readArchitecture() {
    if (readInteger(4) != GGUF_MAGIC)
        throw std::runtime_error("Invalid GGUF magic");
    uint64_t version = readInteger(4);
    if (version != 2 && version != 3)
        throw std::runtime_error("Unsupported GGUF metadata version");
    uint64_t tensorCount = readInteger(8);
    uint64_t metadataCount = readInteger(8);
    if (tensorCount == 0 || metadataCount > remaining / 12)
        throw std::runtime_error("Invalid GGUF metadata header");
    std::string architecture;
    for (uint64_t index = 0; index < metadataCount; ++index) {
        std::string key = readString();
        uint64_t type = readInteger(4);
        if (key == "general.architecture") {
            if (type != 8 || !architecture.empty())
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