#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace ovms {
class GgufMetadataReader {
    std::ifstream file;
    uint64_t remaining;

public:
    explicit GgufMetadataReader(const std::string& path);

    void skip(uint64_t bytes);
    uint64_t readInteger(std::size_t bytes);
    std::string readString();
    void skipValue(uint64_t type);
    std::string readArchitecture();
};
}  // namespace ovms