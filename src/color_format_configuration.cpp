#include "color_format_configuration.hpp"

#include <algorithm>
#include <vector>

namespace ovms {

const char ColorFormatConfiguration::COLOR_FORMAT_DELIMITER = ':';
const std::unordered_map<std::string, ov::preprocess::ColorFormat> ColorFormatConfiguration::colorFormatMap = {
    {"RGB", ov::preprocess::ColorFormat::RGB},
    {"BGR", ov::preprocess::ColorFormat::BGR},
    {"GRAY", ov::preprocess::ColorFormat::GRAY},
    {"NV12", ov::preprocess::ColorFormat::NV12_SINGLE_PLANE},
    {"NV12_2", ov::preprocess::ColorFormat::NV12_TWO_PLANES},
    {"I420", ov::preprocess::ColorFormat::I420_SINGLE_PLANE},
    {"I420_3", ov::preprocess::ColorFormat::I420_THREE_PLANES}
};

Status ColorFormatConfiguration::stringToColorFormat(const std::string& colorFormatStr, ov::preprocess::ColorFormat& colorFormatOut) {
    auto it = colorFormatMap.find(colorFormatStr);
    if (it != colorFormatMap.end()) {
        colorFormatOut = it->second;
        return StatusCode::OK;
    } else {
        SPDLOG_ERROR("Invalid color format value: {}", colorFormatStr);
        return StatusCode::COLOR_FORMAT_WRONG_FORMAT;
    }
}

Status ColorFormatConfiguration::fromString(const std::string& configurationStr, ColorFormatConfiguration& configOut) {
    std::string upperConfigurationStr;
    std::transform(configurationStr.begin(), configurationStr.end(), std::back_inserter(upperConfigurationStr), ::toupper);

    size_t delimiterPos = upperConfigurationStr.find(COLOR_FORMAT_DELIMITER);
    if (delimiterPos == std::string::npos) {
        SPDLOG_ERROR("Invalid color format configuration string: {}", configurationStr);
        return StatusCode::COLOR_FORMAT_WRONG_FORMAT;
    }
    ov::preprocess::ColorFormat targetColorFormat;
    std::string targetColorFormatStr = upperConfigurationStr.substr(0, delimiterPos);

    Status status = stringToColorFormat(targetColorFormatStr, targetColorFormat);
    if (status != StatusCode::OK) {
        return status;
    }

    ov::preprocess::ColorFormat sourceColorFormat;
    std::string sourceColorFormatStr = upperConfigurationStr.substr(delimiterPos + 1);
    status = stringToColorFormat(sourceColorFormatStr, sourceColorFormat);
    if (status != StatusCode::OK) {
        return status;
    }
    configOut = ColorFormatConfiguration(targetColorFormat, sourceColorFormat);

    return StatusCode::OK;
}
} // namespace ovms