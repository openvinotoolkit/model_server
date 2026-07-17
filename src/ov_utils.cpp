//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include "logging.hpp"
#include "profiler.hpp"
#include "status.hpp"
#include "systeminfo.hpp"
#include "tensorinfo.hpp"

namespace ovms {

ov::Tensor createTensorWithNoDataOwnership(ov::element::Type_t precision, const shape_t& shape, void* data) {
    OV_LOGGER("ov::Tensor(precision, shape, data)");
    auto tensor = ov::Tensor(precision, shape, data);
    return tensor;
}

Status createSharedTensor(ov::Tensor& destinationTensor, ov::element::Type_t precision, const ov::Shape& shape) {
    OV_LOGGER("ov::Tensor(precision, shape)");
    destinationTensor = ov::Tensor(precision, shape);
    return StatusCode::OK;
}

std::string getTensorMapString(const std::map<std::string, std::shared_ptr<const TensorInfo>>& inputsInfo) {
    std::stringstream stringStream;
    for (const auto& pair : inputsInfo) {
        const auto& name = pair.first;
        auto inputInfo = pair.second;
        const auto precision = inputInfo->getPrecision();
        const auto& layout = inputInfo->getLayout();
        const auto& shape = inputInfo->getShape();

        stringStream << "\nname: " << name
                     << "; mapping: " << inputInfo->getMappedName()
                     << "; shape: " << shape.toString()
                     << "; precision: " << TensorInfo::getPrecisionAsString(precision)
                     << "; layout: " << TensorInfo::getStringFromLayout(layout);
    }
    return stringStream.str();
}

Status tensorClone(ov::Tensor& destinationTensor, const ov::Tensor& sourceTensor) {
    OVMS_PROFILE_FUNCTION();
    if (sourceTensor.get_element_type() == ov::element::Type_t::string) {
        destinationTensor = ov::Tensor(sourceTensor.get_element_type(), sourceTensor.get_shape());
        const std::string* srcData = sourceTensor.data<std::string>();
        std::string* destData = destinationTensor.data<std::string>();
        for (size_t i = 0; i < sourceTensor.get_shape()[0]; i++) {
            destData[i].assign(srcData[i]);
        }
        return StatusCode::OK;
    }
    OV_LOGGER("ov::Tensor(ov::element::type, shape)");
    destinationTensor = ov::Tensor(sourceTensor.get_element_type(), sourceTensor.get_shape());
    if (destinationTensor.get_byte_size() != sourceTensor.get_byte_size()) {
        SPDLOG_ERROR("tensorClone byte size mismatch destination:{}; source:{}",
            destinationTensor.get_byte_size(),
            sourceTensor.get_byte_size());
        return StatusCode::OV_CLONE_TENSOR_ERROR;
    }
    std::memcpy(destinationTensor.data(), sourceTensor.data(), sourceTensor.get_byte_size());
    return StatusCode::OK;
}

std::optional<ov::Layout> getLayoutFromRTMap(const ov::RTMap& rtMap) {
    OV_LOGGER("const auto& [k, v] : ov::RTMap& rtMap");
    for (const auto& [k, v] : rtMap) {
        try {
            OV_LOGGER("v.as<ov::LayoutAttribute>().value");
            return v.as<ov::LayoutAttribute>().value;
        } catch (ov::Exception&) {
        }
    }
    return std::nullopt;
}

static void insertSupportedKeys(std::set<std::string>& aggregatedPluginSupportedConfigKeys, const std::string& pluginName, const ov::Core& ieCore) {
    auto prop = ov::supported_properties;
    try {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Validating plugin: {}; configuration", pluginName);
        OV_LOGGER("ov::Core: {}, ieCore->get_property({}, ov::supported_properties)", reinterpret_cast<const void*>(&ieCore), pluginName);
        std::vector<ov::PropertyName> pluginSupportedConfigKeys = ieCore.get_property(pluginName, prop);
        std::set<std::string> pluginSupportedConfigKeysSet(pluginSupportedConfigKeys.begin(), pluginSupportedConfigKeys.end());
        aggregatedPluginSupportedConfigKeys.insert(pluginSupportedConfigKeys.begin(), pluginSupportedConfigKeys.end());
    } catch (std::exception& e) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value. Error: {}", pluginName, prop.name(), e.what());
    } catch (...) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Exception thrown from IE when requesting plugin: {}; key: {}; value.", pluginName, prop.name());
    }
}

Status validatePluginConfiguration(const plugin_config_t& pluginConfig, const std::string& targetDevice, const ov::Core& ieCore) {
    std::set<std::string> pluginSupportedConfigKeys;
    std::string pluginDelimiter = ":";
    auto pluginDelimeterPos = targetDevice.find(pluginDelimiter);
    if (pluginDelimeterPos != std::string::npos) {
        std::string pluginName = targetDevice.substr(0, pluginDelimeterPos);
        insertSupportedKeys(pluginSupportedConfigKeys, pluginName, ieCore);
        char deviceDelimiter = ',';
        std::stringstream ss(targetDevice.substr(pluginDelimeterPos + 1, targetDevice.length()));
        std::string deviceName;

        while (getline(ss, deviceName, deviceDelimiter)) {
            char bracket = '(';
            auto bracketPos = deviceName.find(bracket);
            if (bracketPos != std::string::npos) {
                deviceName = deviceName.substr(0, bracketPos);
            }
            insertSupportedKeys(pluginSupportedConfigKeys, deviceName, ieCore);
        }
    } else {
        insertSupportedKeys(pluginSupportedConfigKeys, targetDevice, ieCore);
    }

    pluginSupportedConfigKeys.insert("ENABLE_MMAP");  // WA: always supported

    for (auto& config : pluginConfig) {
        if (std::find(pluginSupportedConfigKeys.begin(), pluginSupportedConfigKeys.end(), config.first) == pluginSupportedConfigKeys.end()) {
            SPDLOG_LOGGER_ERROR(modelmanager_logger, "Plugin config key: {} not found in supported config keys for device: {}.", config.first, targetDevice);
            SPDLOG_LOGGER_INFO(modelmanager_logger, "List of supported keys for this device:");
            for (std::string supportedKey : pluginSupportedConfigKeys) {
                SPDLOG_LOGGER_INFO(modelmanager_logger, "{}", supportedKey);
            }
            return StatusCode::MODEL_CONFIG_INVALID;
        }
    }

    return StatusCode::OK;
}

std::string recommendTargetDevice(const std::vector<GpuDeviceInfo>& gpuDevices) {
    if (gpuDevices.empty()) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "No GPU devices found, recommending CPU");
        return "CPU";
    }

    std::vector<GpuDeviceInfo> discreteGpus;
    std::vector<GpuDeviceInfo> integratedGpus;
    for (const auto& gpu : gpuDevices) {
        if (gpu.isDiscrete) {
            discreteGpus.push_back(gpu);
        } else {
            integratedGpus.push_back(gpu);
        }
    }

    if (discreteGpus.size() == 1) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Single discrete GPU found, recommending: {}", discreteGpus[0].name);
        return discreteGpus[0].name;
    }

    if (discreteGpus.size() > 1) {
        auto best = std::max_element(discreteGpus.begin(), discreteGpus.end(),
            [](const GpuDeviceInfo& a, const GpuDeviceInfo& b) {
                return a.freeMemBytes < b.freeMemBytes;
            });
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Multiple discrete GPUs found, recommending {} with {} bytes free VRAM", best->name, best->freeMemBytes);
        return best->name;
    }

    if (!integratedGpus.empty()) {
        SPDLOG_LOGGER_INFO(modelmanager_logger, "Integrated GPU found, recommending: {}", integratedGpus[0].name);
        return integratedGpus[0].name;
    }

    SPDLOG_LOGGER_INFO(modelmanager_logger, "Falling back to CPU");
    return "CPU";
}

std::string recommendTargetDevice() {
    static ov::Core core;
    auto availableDevices = core.get_available_devices();

    std::vector<GpuDeviceInfo> gpuDevices;

    for (const auto& device : availableDevices) {
        if (device.find("GPU") != 0) {
            continue;
        }

        GpuDeviceInfo info;
        info.name = device;

        try {
            auto deviceType = core.get_property(device, ov::device::type);
            info.isDiscrete = (deviceType == ov::device::Type::DISCRETE);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Failed to get DEVICE_TYPE for {}: {}; treating as integrated/unknown GPU", device, e.what());
            info.isDiscrete = false;
        }

        try {
            uint64_t totalMem = core.get_property(device, "GPU_DEVICE_TOTAL_MEM_SIZE").as<uint64_t>();
            uint64_t usedMem = 0;
            try {
                auto memStats = core.get_property(device, "GPU_MEMORY_STATISTICS").as<std::map<std::string, uint64_t>>();
                auto it = memStats.find("usm_device");
                if (it != memStats.end()) {
                    usedMem = it->second;
                }
            } catch (...) {
                // Memory statistics may not be available before any model is loaded
            }
            info.freeMemBytes = static_cast<int64_t>(totalMem) - static_cast<int64_t>(usedMem);
        } catch (const std::exception& e) {
            SPDLOG_LOGGER_WARN(modelmanager_logger, "Failed to get memory info for {}: {}", device, e.what());
            info.freeMemBytes = 0;
        }

        gpuDevices.push_back(info);
    }

    return recommendTargetDevice(gpuDevices);
}

Status applyDefaultCpuProperties(ov::AnyMap& properties, uint16_t coreCount, uint16_t physicalCoresPerSocket, uint16_t socketsCount, uint16_t dockerCpuQuota) {
    if (properties.find(ov::hint::enable_cpu_pinning.name()) == properties.end()) {
        const bool cpuPinning = dockerCpuQuota <= 0;
        properties[ov::hint::enable_cpu_pinning.name()] = cpuPinning;
        SPDLOG_DEBUG("applyDefaultCpuProperties, DockerCPUQuota: {} - setting enable_cpu_pinning to {}", dockerCpuQuota, cpuPinning);
    }

    bool isThroughput = false;
    const auto perfIt = properties.find(ov::hint::performance_mode.name());
    if (perfIt != properties.end()) {
        try {
            isThroughput = (perfIt->second.as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::THROUGHPUT);
        } catch (...) {
            try {
                isThroughput = (perfIt->second.as<std::string>() == "THROUGHPUT");
            } catch (...) {
            }
        }
        if (isThroughput && properties.find(ov::num_streams.name()) == properties.end()) {
            int numStreams = std::min(static_cast<int>(coreCount), static_cast<int>(physicalCoresPerSocket * socketsCount));
            properties[ov::num_streams.name()] = numStreams;
            SPDLOG_DEBUG("applyDefaultCpuProperties, CoreCount: {}, PhysicalCoresPerSocket: {}, SocketsCount: {} - setting num_streams to {} (THROUGHPUT hint active)", coreCount, physicalCoresPerSocket, socketsCount, numStreams);
        }
    }

    if (properties.find(ov::inference_num_threads.name()) == properties.end()) {
        if (coreCount <= physicalCoresPerSocket * socketsCount) {
            int numThreads = static_cast<int>(coreCount);
            properties[ov::inference_num_threads.name()] = numThreads;
            SPDLOG_DEBUG("applyDefaultCpuProperties: CoreCount: {}, PhysicalCoresPerSocket: {}, SocketsCount: {}, setting inference_num_threads to {}", coreCount, physicalCoresPerSocket, socketsCount, numThreads);
        }
    }
    return StatusCode::OK;
}

Status applyDefaultCpuProperties(ov::AnyMap& properties) {
#ifdef __linux__
    try {
        if (!isRunningInDocker()) {
            return StatusCode::OK;
        }
        return applyDefaultCpuProperties(properties, getCoreCount(), getPhysicalCoresPerSocket(), getSocketsCount(), getDockerCpuQuota());
    } catch (const std::exception& ex) {
        SPDLOG_WARN("Exception while applying default CPU properties: {}", ex.what());
    } catch (...) {
        SPDLOG_WARN("Unknown exception while applying default CPU properties");
    }
#endif
    return StatusCode::OK;
}

}  // namespace ovms
