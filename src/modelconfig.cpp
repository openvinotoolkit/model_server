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
#include "modelconfig.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "filesystem.hpp"
#include "logging.hpp"
#include "model_version_policy.hpp"
#include "schema.hpp"
#include "stringutils.hpp"

namespace ovms {
ModelConfig::ModelConfig(const std::string& name,
    const std::string& basePath,
    const std::string& targetDevice,
    const std::string& configBatchSize,
    uint64_t nireq,
    bool stateful,
    bool idleSequenceCleanup,
    bool lowLatencyTransformation,
    uint32_t maxSequenceNumber,
    const std::string& cacheDir,
    model_version_t version,
    const std::string& localPath) :
    name(name),
    basePath(basePath),
    localPath(localPath),
    targetDevice(targetDevice),
    modelVersionPolicy(ModelVersionPolicy::getDefaultVersionPolicy()),
    nireq(nireq),
    stateful(stateful),
    idleSequenceCleanup(idleSequenceCleanup),
    lowLatencyTransformation(lowLatencyTransformation),
    maxSequenceNumber(maxSequenceNumber),
    cacheDir(cacheDir),
    version(version),
    pluginConfig({}),
    layout(""),
    shapes({}),
    layouts({}),
    mappingInputs({}),
    mappingOutputs({}) {
    setBatchingParams(configBatchSize);
}

const std::string ANONYMOUS_INPUT_NAME = "ANONYMOUS_INPUT_NAME";
const std::string MAPPING_CONFIG_JSON = "mapping_config.json";

bool ModelConfig::isDeviceUsed(const std::string& device) const {
    if (this->isSingleDeviceUsed(device))
        return true;
    if (targetDevice.find(device) != std::string::npos)
        return true;
    if (targetDevice == "AUTO")
        return true;
    return false;
}

bool ModelConfig::isReloadRequired(const ModelConfig& rhs) const {
    if (this->name != rhs.name) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to name mismatch", this->name);
        return true;
    }
    if (this->stateful != rhs.stateful) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to stateful mismatch", this->name);
        return true;
    }
    if (this->idleSequenceCleanup != rhs.idleSequenceCleanup) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to idleSequenceCleanup mismatch", this->name);
        return true;
    }
    if (this->maxSequenceNumber != rhs.maxSequenceNumber) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to maxSequenceNumber mismatch", this->name);
        return true;
    }
    if (this->lowLatencyTransformation != rhs.lowLatencyTransformation) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to lowLatencyTransformation mismatch", this->name);
        return true;
    }
    if (this->basePath != rhs.basePath) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to original base path mismatch", this->name);
        return true;
    }
    if (this->targetDevice != rhs.targetDevice) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to target device mismatch", this->name);
        return true;
    }
    if (this->batchingMode != rhs.batchingMode) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to batching mode mismatch", this->name);
        return true;
    }
    if (!isBatchSizeConfigurationEqual(rhs)) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to batch size mismatch", this->name);
        return true;
    }
    if (this->nireq != rhs.nireq) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to nireq mismatch", this->name);
        return true;
    }
    if (this->pluginConfig != rhs.pluginConfig) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to plugin config mismatch", this->name);
        return true;
    }
    if (!isLayoutConfigurationEqual(rhs)) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to named layout mismatch", this->name);
        return true;
    }
    if (!isShapeConfigurationEqual(rhs)) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to shape configuration mismatch", this->name);
        return true;
    }
    if (isCustomLoaderConfigChanged(rhs)) {
        return true;
    }
    if (this->isAllowCacheSetToTrue() != rhs.isAllowCacheSetToTrue()) {
        return true;
    }
    return false;
}

bool ModelConfig::isCustomLoaderConfigChanged(const ModelConfig& rhs) const {
    if (this->customLoaderOptionsConfigMap.size() != rhs.customLoaderOptionsConfigMap.size()) {
        SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to custom loader config mismatch", this->name);
        return true;
    }
    if (this->customLoaderOptionsConfigMap.size() > 0 && rhs.customLoaderOptionsConfigMap.size() > 0) {
        if (!(this->customLoaderOptionsConfigMap == rhs.customLoaderOptionsConfigMap)) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "ModelConfig {} reload required due to custom loader config mismatch", this->name);
            return true;
        }
    }
    return false;
}

bool ModelConfig::isBatchSizeConfigurationEqual(const ModelConfig& rhs) const {
    if (this->batchSize.has_value() != rhs.batchSize.has_value()) {
        return false;
    }
    if (this->batchSize.has_value() && this->batchSize.value() != rhs.batchSize.value()) {
        return false;
    }
    return true;
}

bool ModelConfig::isLayoutConfigurationEqual(const ModelConfig& rhs) const {
    if (this->layout != rhs.layout) {
        return false;
    }

    if (this->layouts.size() != rhs.layouts.size()) {
        return false;
    }
    for (const auto& [name, layoutConfig] : this->layouts) {
        auto it = rhs.layouts.find(name);
        if (it == rhs.layouts.end()) {
            return false;
        }
        if (layoutConfig != it->second) {
            return false;
        }
    }
    return true;
}

bool ModelConfig::isShapeConfigurationEqual(const ModelConfig& rhs) const {
    if (this->shapes.size() != rhs.shapes.size()) {
        return false;
    }
    for (const auto& [name, shape] : this->shapes) {
        auto it = rhs.shapes.find(name);
        if (it == rhs.shapes.end()) {
            return false;
        }
        if (shape != it->second) {
            return false;
        }
    }
    return true;
}

std::tuple<Mode, std::optional<Dimension>> ModelConfig::extractBatchingParams(std::string configBatchSize) {
    Mode batchingMode = FIXED;
    std::optional<Dimension> effectiveBatchSize = std::nullopt;
    if (configBatchSize == "auto") {
        batchingMode = AUTO;
    } else if (configBatchSize == "0") {
        // do nothing
    } else {
        Dimension dim;
        auto status = Dimension::fromString(configBatchSize, dim);
        if (!status.ok()) {
            SPDLOG_WARN("Wrong batch size parameter provided. Model batch size will be set to default.");
        } else {
            effectiveBatchSize = dim;
        }
    }
    return std::tuple<Mode, std::optional<Dimension>>{batchingMode, effectiveBatchSize};
}

Status ModelConfig::parseModelVersionPolicy(std::string command) {
    rapidjson::Document node;
    if (command == "") {
        modelVersionPolicy = ModelVersionPolicy::getDefaultVersionPolicy();
        return StatusCode::OK;
    }

    if (node.Parse(command.c_str()).HasParseError()) {
        return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
    }

    if (!node.IsObject()) {
        return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
    }
    if (node.MemberCount() != 1) {
        return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
    }

    auto m = node.FindMember("all");
    if (m != node.MemberEnd()) {
        modelVersionPolicy = std::make_shared<AllModelVersionPolicy>();
        return StatusCode::OK;
    }

    m = node.FindMember("specific");
    if (m != node.MemberEnd()) {
        auto& specific = m->value;
        if (specific.MemberCount() != 1) {
            return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
        }
        m = specific.FindMember("versions");
        if (m == specific.MemberEnd()) {
            return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
        }
        std::vector<model_version_t> versions;
        for (auto& version : m->value.GetArray()) {
            if (version.IsUint64() && version.GetUint64() > 0) {
                versions.push_back(version.GetUint64());
            } else {
                SPDLOG_WARN("Model policy specified in config contains invalid version. Version should be a number greater than 0.");
            }
        }
        modelVersionPolicy = std::make_shared<SpecificModelVersionPolicy>(versions);
        return StatusCode::OK;
    }

    m = node.FindMember("latest");
    if (m != node.MemberEnd()) {
        auto& latest = m->value;
        if (latest.MemberCount() != 1) {
            return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
        }
        m = latest.FindMember("num_versions");
        if (m == latest.MemberEnd()) {
            return StatusCode::MODEL_VERSION_POLICY_WRONG_FORMAT;
        }
        modelVersionPolicy = std::make_shared<LatestModelVersionPolicy>(m->value.GetInt64());
        return StatusCode::OK;
    }

    return StatusCode::MODEL_VERSION_POLICY_UNSUPPORTED_KEY;
}

Status ModelConfig::parsePluginConfig(const rapidjson::Value& node) {
    if (!node.IsObject()) {
        return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
    }

    for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
        if (it->value.IsString()) {
            if (((it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS")) && (it->value.GetString() == std::string("CPU_THROUGHPUT_AUTO"))) || ((it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) && (it->value.GetString() == std::string("GPU_THROUGHPUT_AUTO")))) {
                pluginConfig["PERFORMANCE_HINT"] = "THROUGHPUT";
                SPDLOG_WARN("{} plugin config key is deprecated. Use PERFORMANCE_HINT instead", it->name.GetString());
            } else {
                if ((it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS")) || (it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS"))) {
                    pluginConfig["NUM_STREAMS"] = it->value.GetString();
                    SPDLOG_WARN("{} plugin config key is deprecated. Use NUM_STREAMS instead", it->name.GetString());
                } else if (it->name.GetString() == std::string("CPU_BIND_THREAD")) {
                    if (it->value.GetString() == std::string("YES")) {
                        pluginConfig["AFFINITY"] = "CORE";
                        SPDLOG_WARN("{} plugin config key is deprecated. Use AFFINITY instead", it->name.GetString());
                    } else if (it->value.GetString() == std::string("NO")) {
                        pluginConfig["AFFINITY"] = "NONE";
                        SPDLOG_WARN("{} plugin config key is deprecated. Use AFFINITY instead", it->name.GetString());
                    } else {
                        SPDLOG_ERROR("{} plugin config key has invalid value and is deprecated. Use AFFINITY key instead", it->name.GetString());
                        return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
                    }
                } else if (it->name.GetString() == std::string("CPU_THREADS_NUM")) {
                    pluginConfig["INFERENCE_NUM_THREADS"] = it->value.GetString();
                    SPDLOG_WARN("{} plugin config key is deprecated. Use INFERENCE_NUM_THREADS instead", it->name.GetString());
                } else {
                    pluginConfig[it->name.GetString()] = it->value.GetString();
                }
            }

        } else if (it->value.IsInt64()) {
            if (it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS") || it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) {
                pluginConfig["NUM_STREAMS"] = std::to_string(it->value.GetInt64());
                SPDLOG_WARN("{} plugin config key is deprecated. Use  NUM_STREAMS instead", it->name.GetString());
            } else {
                pluginConfig[it->name.GetString()] = std::to_string(it->value.GetInt64());
            }
        } else if (it->value.IsDouble()) {
            if (it->name.GetString() == std::string("CPU_THROUGHPUT_STREAMS") || it->name.GetString() == std::string("GPU_THROUGHPUT_STREAMS")) {
                pluginConfig["NUM_STREAMS"] = std::to_string(it->value.GetDouble());
                SPDLOG_WARN("{} plugin config key is deprecated. Use  NUM_STREAMS instead", it->name.GetString());
            } else {
                pluginConfig[it->name.GetString()] = std::to_string(it->value.GetDouble());
            }
        } else {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
        }
    }

    return StatusCode::OK;
}

Status ModelConfig::parseShapeParameter(const rapidjson::Value& node) {
    if (!node.IsObject()) {
        return StatusCode::SHAPE_WRONG_FORMAT;
    }

    shapes_info_map_t shapes;
    for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
        if (!it->value.IsString()) {
            return StatusCode::SHAPE_WRONG_FORMAT;
        }
        ShapeInfo shapeInfo;
        auto status = parseShape(shapeInfo, it->value.GetString());
        if (!status.ok()) {
            return status;
        }
        shapes[it->name.GetString()] = shapeInfo;
    }
    this->shapes = shapes;

    return StatusCode::OK;
}

Status ModelConfig::parseShapeParameter(const std::string& command) {
    this->shapes.clear();

    if (command.empty()) {
        return StatusCode::OK;
    }

    // parse as string
    if (command.front() == shapeLeft || command == "auto") {
        ShapeInfo shapeInfo;
        auto status = parseShape(shapeInfo, command);
        if (!status.ok()) {
            return status;
        }
        this->addShape(ANONYMOUS_INPUT_NAME, shapeInfo);
        return StatusCode::OK;
    }

    // parse as json
    rapidjson::Document node;
    if (command.empty()) {
        return StatusCode::OK;
    }
    if (node.Parse(command.c_str()).HasParseError()) {
        return StatusCode::SHAPE_WRONG_FORMAT;
    }
    return parseShapeParameter(node);
}

Status ModelConfig::parseLayoutParameter(const rapidjson::Value& node) {
    if (!node.IsObject()) {
        return StatusCode::LAYOUT_WRONG_FORMAT;
    }
    layout_configurations_map_t layouts;
    for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
        if (!it->value.IsString()) {
            return StatusCode::LAYOUT_WRONG_FORMAT;
        }
        std::string layoutStr = it->value.GetString();
        std::transform(layoutStr.begin(), layoutStr.end(), layoutStr.begin(), ::toupper);

        LayoutConfiguration layout;
        auto status = LayoutConfiguration::fromString(layoutStr, layout);
        if (!status.ok()) {
            return status;
        }
        layouts[it->name.GetString()] = layout;
    }
    setLayouts(layouts);

    return StatusCode::OK;
}

Status ModelConfig::parseLayoutParameter(const std::string& command) {
    this->layouts.clear();
    this->layout = LayoutConfiguration();
    if (command.empty()) {
        return StatusCode::OK;
    }

    std::string upperCaseCommand;
    std::transform(command.begin(), command.end(), std::back_inserter(upperCaseCommand), ::toupper);

    erase_spaces(upperCaseCommand);

    if (*upperCaseCommand.begin() != '{') {
        LayoutConfiguration layout;
        auto status = LayoutConfiguration::fromString(upperCaseCommand, layout);
        if (!status.ok()) {
            return status;
        }
        setLayout(layout);
        return StatusCode::OK;
    }

    // parse as json
    rapidjson::Document node;
    if (node.Parse(command.c_str()).HasParseError()) {
        return StatusCode::LAYOUT_WRONG_FORMAT;
    }
    return parseLayoutParameter(node);
}

Status ModelConfig::parseShape(ShapeInfo& shapeInfo, const std::string& str) {
    if (str == "auto") {
        shapeInfo.shapeMode = AUTO;
        return StatusCode::OK;
    }

    shapeInfo.shapeMode = FIXED;
    shapeInfo.shape = Shape();
    return Shape::fromString(str, shapeInfo.shape);
}

Status ModelConfig::parseModelMapping() {
    SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Parsing model: {} mapping from path: {}", getName(), getPath());
    mappingInputs.clear();
    mappingOutputs.clear();
    std::filesystem::path path = this->getPath();
    path.append(MAPPING_CONFIG_JSON);

    std::ifstream ifs(path.c_str());
    if (!ifs.good()) {
        return StatusCode::FILE_INVALID;
    }

    rapidjson::Document doc;
    rapidjson::IStreamWrapper isw(ifs);
    if (doc.ParseStream(isw).HasParseError()) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model: {} mapping configuration file is not a valid JSON file.", getName());
        return StatusCode::JSON_INVALID;
    }

    if (validateJsonAgainstSchema(doc, MODELS_MAPPING_SCHEMA) != StatusCode::OK) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Model: {} mapping configuration file is not a valid JSON file.", getName());
        return StatusCode::JSON_INVALID;
    }
    auto itr = doc.FindMember("inputs");
    if (itr == doc.MemberEnd()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Couldn't load inputs object from file {}", path.c_str());
    } else {
        // Process inputs
        for (const auto& key : itr->value.GetObject()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Loaded input mapping {} => {}", key.name.GetString(), key.value.GetString());
            mappingInputs[key.name.GetString()] = key.value.GetString();
            reversedMappingInputs[key.value.GetString()] = key.name.GetString();
        }
    }
    itr = doc.FindMember("outputs");
    if (itr == doc.MemberEnd()) {
        SPDLOG_LOGGER_WARN(modelmanager_logger, "Couldn't load outputs object from file {}", path.c_str());
    } else {
        // Process outputs
        const auto it = doc.FindMember("outputs");
        for (const auto& key : it->value.GetObject()) {
            SPDLOG_LOGGER_DEBUG(modelmanager_logger, "Loaded output mapping {} => {}", key.name.GetString(), key.value.GetString());
            mappingOutputs[key.name.GetString()] = key.value.GetString();
            reversedMappingOutputs[key.value.GetString()] = key.name.GetString();
        }
    }
    return StatusCode::OK;
}

Status ModelConfig::parseNode(const rapidjson::Value& v) {
    this->setName(v["name"].GetString());
    try {
        this->setBasePath(v["base_path"].GetString());
    } catch (std::logic_error& e) {
        SPDLOG_DEBUG("Relative path error: {}", e.what());
        return StatusCode::INTERNAL_ERROR;
    }
    Status firstErrorStatus = StatusCode::OK;

    // Check for optional parameters
    if (v.HasMember("batch_size")) {
        if (v["batch_size"].IsString()) {
            this->setBatchingParams(v["batch_size"].GetString());
        } else {
            this->setBatchingParams(v["batch_size"].GetUint64());
        }
    }
    if (v.HasMember("target_device"))
        this->setTargetDevice(v["target_device"].GetString());
    if (v.HasMember("version")) {
        this->setVersion(v["version"].GetUint64());
    }
    if (v.HasMember("nireq"))
        this->setNireq(v["nireq"].GetUint64());

    if (v.HasMember("shape")) {
        // Legacy format as string
        if (v["shape"].IsString()) {
            ShapeInfo shapeInfo;
            auto status = parseShape(shapeInfo, v["shape"].GetString());
            if (!status.ok()) {
                if (!firstErrorStatus.ok()) {
                    firstErrorStatus = status;
                }
                SPDLOG_WARN("There was an error parsing shape {}", v["shape"].GetString());
            } else {
                this->addShape(ANONYMOUS_INPUT_NAME, shapeInfo);
            }
        } else {
            // Map of shapes
            for (auto& s : v["shape"].GetObject()) {
                ShapeInfo shapeInfo;
                bool valid = true;
                // check if legacy format is used
                if (s.value.IsString()) {
                    auto status = ModelConfig::parseShape(shapeInfo, s.value.GetString());
                    if (!status.ok()) {
                        if (!firstErrorStatus.ok()) {
                            firstErrorStatus = status;
                        }
                        SPDLOG_WARN("There was an error parsing shape {}", s.name.GetString());
                        valid = false;
                    }
                } else {
                    for (auto& sh : s.value.GetArray()) {
                        if (sh.IsUint64()) {
                            size_t dim = sh.GetUint64();
                            if (dim > static_cast<size_t>(std::numeric_limits<dimension_value_t>::max())) {
                                SPDLOG_WARN("There was an error parsing shape {}", s.name.GetString());
                                if (!firstErrorStatus.ok()) {
                                    firstErrorStatus = StatusCode::SHAPE_WRONG_FORMAT;
                                }
                                valid = false;
                                break;
                            }
                            shapeInfo.shape.add(Dimension{static_cast<dimension_value_t>(dim)});
                        } else {
                            SPDLOG_WARN("There was an error parsing shape {}", s.name.GetString());
                            if (!firstErrorStatus.ok()) {
                                firstErrorStatus = StatusCode::SHAPE_WRONG_FORMAT;
                            }
                            valid = false;
                            break;
                        }
                    }
                }
                if (s.name.GetString() != ANONYMOUS_INPUT_NAME) {
                    if (valid) {
                        this->addShape(s.name.GetString(), shapeInfo);
                    }
                } else {
                    SPDLOG_WARN("Provided shape name: {} is forbidden and will be omitted", ANONYMOUS_INPUT_NAME);
                }
            }
        }
    }

    if (v.HasMember("layout")) {
        if (v["layout"].IsString()) {
            Status status = this->parseLayoutParameter(v["layout"].GetString());
            if (!status.ok()) {
                return status;
            }
        } else {
            Status status = this->parseLayoutParameter(v["layout"]);
            if (!status.ok()) {
                return status;
            }
        }
    }

    if (v.HasMember("plugin_config")) {
        auto status = parsePluginConfig(v["plugin_config"]);
        if (!status.ok()) {
            if (!firstErrorStatus.ok()) {
                firstErrorStatus = status;
            }
            SPDLOG_ERROR("Couldn't parse plugin config");
            return status;
        }
    }

    if (v.HasMember("stateful"))
        this->setStateful(v["stateful"].GetBool());

    if (v.HasMember("low_latency_transformation")) {
        if (!this->isStateful()) {
            SPDLOG_ERROR("Low latency transformation parameter was set for non stateful model {}.", v["name"].GetString());
            return StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER;
        }
        this->setLowLatencyTransformation(v["low_latency_transformation"].GetBool());
    }

    if (v.HasMember("idle_sequence_cleanup")) {
        if (!this->isStateful()) {
            SPDLOG_ERROR("Idle sequence cleanup parameter was set for non stateful model {}.", v["name"].GetString());
            return StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER;
        }
        this->setIdleSequenceCleanup(v["idle_sequence_cleanup"].GetBool());
    }

    if (v.HasMember("max_sequence_number")) {
        if (!this->isStateful()) {
            SPDLOG_ERROR("Max sequence number parameter was set for non stateful model {}.", v["name"].GetString());
            return StatusCode::INVALID_NON_STATEFUL_MODEL_PARAMETER;
        }
        if (!v["max_sequence_number"].IsUint()) {
            SPDLOG_ERROR("Sequence maximum number parameter was set above unsigned int value for model {}.", v["name"].GetString());
            return StatusCode::INVALID_MAX_SEQUENCE_NUMBER;
        }
        this->setMaxSequenceNumber(v["max_sequence_number"].GetUint());
    }

    if (v.HasMember("model_version_policy")) {
        rapidjson::StringBuffer buffer;
        buffer.Clear();
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        v["model_version_policy"].Accept(writer);
        const auto& status = parseModelVersionPolicy(buffer.GetString());
        if (!status.ok()) {
            if (!firstErrorStatus.ok()) {
                firstErrorStatus = status;
            }
            SPDLOG_WARN("Couldn't parse model version policy. {}", status.string());
        }
    } else {
        modelVersionPolicy = ModelVersionPolicy::getDefaultVersionPolicy();
    }

    SPDLOG_DEBUG("Specified model parameters:");
    SPDLOG_DEBUG("model_basepath: {}", getBasePath());
    SPDLOG_DEBUG("model_name: {}", getName());
    SPDLOG_DEBUG("batch_size: {}", getBatchSize().has_value() ? getBatchSize().value().toString() : "not configured");
    if (isShapeAnonymous()) {
        SPDLOG_DEBUG("shape: {}", std::string(getShapes().begin()->second));
    } else {
        SPDLOG_DEBUG("shape:");
        for (auto& [shapeInput, shapeValue] : getShapes()) {
            SPDLOG_DEBUG("  {}: {}", shapeInput, std::string(shapeValue));
        }
    }
    if (getModelVersionPolicy()) {
        SPDLOG_DEBUG("model_version_policy: {}", std::string(*getModelVersionPolicy()));
    }
    SPDLOG_DEBUG("nireq: {}", getNireq());
    SPDLOG_DEBUG("target_device: {}", getTargetDevice());
    SPDLOG_DEBUG("plugin_config:");
    for (auto& [pluginParameter, pluginValue] : getPluginConfig()) {
        SPDLOG_DEBUG("  {}: {}", pluginParameter, pluginValue.as<std::string>());
    }

    bool batchSizeSet = (getBatchingMode() != FIXED || getBatchSize().has_value());
    bool shapeSet = (getShapes().size() > 0);

    SPDLOG_DEBUG("Batch size set: {}, shape set: {}", batchSizeSet, shapeSet);
    if (batchSizeSet && shapeSet) {
        SPDLOG_WARN("Both shape and batch size have been defined. Batch size parameter will be ignored.");
        setBatchingMode(FIXED);
        setBatchSize(std::nullopt);
    }

    SPDLOG_DEBUG("stateful: {}", isStateful());
    if (isStateful()) {
        SPDLOG_DEBUG("idle_sequence_cleanup: {}", getIdleSequenceCleanup());
        SPDLOG_DEBUG("max_sequence_number: {}", getMaxSequenceNumber());
        SPDLOG_DEBUG("low_latency_transformation: {}", isLowLatencyTransformationUsed());
    }

    // Model Cache options
    if (v.HasMember("allow_cache")) {
        setAllowCache(v["allow_cache"].GetBool());
        SPDLOG_DEBUG("allow_cache: {}", v["allow_cache"].GetBool());
    }

    // if the config has models which require custom loader to be used, then load the same here
    if (v.HasMember("custom_loader_options")) {
        if (!parseCustomLoaderOptionsConfig(v["custom_loader_options"]).ok()) {
            SPDLOG_ERROR("Couldn't parse custom loader options config");
        }
    }
    return StatusCode::OK;
}

Status ModelConfig::parseCustomLoaderOptionsConfig(const rapidjson::Value& node) {
    if (!node.IsObject()) {
        return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
    }
    for (auto it = node.MemberBegin(); it != node.MemberEnd(); ++it) {
        if (!it->value.IsString()) {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
        }
        addCustomLoaderOption(it->name.GetString(), it->value.GetString());
    }
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    node.Accept(writer);
    customLoaderOptionsStr = buffer.GetString();

    return StatusCode::OK;
}

std::string ModelConfig::layoutConfigurationToString() const {
    if (getLayout().isSet()) {
        return getLayout().toString();
    }
    std::stringstream ss;
    for (const auto& [name, layoutCfg] : getLayouts()) {
        ss << name << " " << layoutCfg.toString() << "; ";
    }
    return ss.str();
}
void ModelConfig::setBasePath(const std::string& basePath) {
    FileSystem::setPath(this->basePath, basePath, this->rootDirectoryPath);
}

}  // namespace ovms
