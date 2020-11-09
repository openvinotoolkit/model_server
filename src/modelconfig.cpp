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
#include <sstream>

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "schema.hpp"
#include "stringutils.hpp"

namespace ovms {

bool ModelConfig::isReloadRequired(const ModelConfig& rhs) const {
    if (this->name != rhs.name) {
        spdlog::debug("ModelConfig {} reload required due to name mismatch", this->name);
        return true;
    }
    if (this->basePath != rhs.basePath) {
        spdlog::debug("ModelConfig {} reload required due to original base path mismatch", this->name);
        return true;
    }
    if (this->targetDevice != rhs.targetDevice) {
        spdlog::debug("ModelConfig {} reload required due to target device mismatch", this->name);
        return true;
    }
    if (this->batchingMode != rhs.batchingMode) {
        spdlog::debug("ModelConfig {} reload required due to batching mode mismatch", this->name);
        return true;
    }
    if (this->batchSize != rhs.batchSize) {
        spdlog::debug("ModelConfig {} reload required due to batch size mismatch", this->name);
        return true;
    }
    if (this->nireq != rhs.nireq) {
        spdlog::debug("ModelConfig {} reload required due to nireq mismatch", this->name);
        return true;
    }
    if (this->pluginConfig != rhs.pluginConfig) {
        spdlog::debug("ModelConfig {} reload required due to plugin config mismatch", this->name);
        return true;
    }
    if (this->layout != rhs.layout) {
        spdlog::debug("ModelConfig {} reload required due to no named layout mismatch", this->name);
        return true;
    }
    if (this->layouts != rhs.layouts) {
        spdlog::debug("ModelConfig {} reload required due to named layout mismatch", this->name);
        return true;
    }
    if (!isShapeConfigurationEqual(rhs)) {
        spdlog::debug("ModelConfig {} reload required due to shape configuration mismatch", this->name);
        return true;
    }

    return false;
}

bool ModelConfig::isShapeConfigurationEqual(const ModelConfig& rhs) const {
    if (this->shapes.size() != rhs.shapes.size()) {
        return false;
    }
    if (this->shapes.size() > 1) {
        return this->shapes == rhs.shapes;
    }
    if (this->shapes.size() == 1) {
        if (this->shapes.begin()->first != rhs.shapes.begin()->first) {
            return false;
        } else {
            return this->shapes.begin()->second == rhs.shapes.begin()->second;
        }
    }
    return true;
}

std::tuple<Mode, size_t> ModelConfig::extractBatchingParams(std::string configBatchSize) {
    Mode batchingMode = FIXED;
    size_t effectiveBatchSize = 0;
    if (configBatchSize == "auto") {
        batchingMode = AUTO;
    } else {
        try {
            effectiveBatchSize = std::stoi(configBatchSize);
        } catch (const std::invalid_argument& e) {
            SPDLOG_ERROR("Wrong batch size parameter provided. Model batch size will be set to default.");
        }
    }
    return std::tuple<Mode, size_t>{batchingMode, effectiveBatchSize};
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
                spdlog::warn("Model policy specified in config contains invalid version. Version should be a number greater than 0.");
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
        if (!it->value.IsString()) {
            return StatusCode::PLUGIN_CONFIG_WRONG_FORMAT;
        }
        pluginConfig[it->name.GetString()] = it->value.GetString();
    }

    return StatusCode::OK;
}

Status ModelConfig::parseShapeParameter(const rapidjson::Value& node) {
    if (!node.IsObject()) {
        return StatusCode::SHAPE_WRONG_FORMAT;
    }

    shapes_map_t shapes;
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
    setShapes(shapes);

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

Status ModelConfig::parseShape(ShapeInfo& shapeInfo, const std::string& str) {
    if (str == "auto") {
        shapeInfo.shapeMode = AUTO;
        return StatusCode::OK;
    }

    std::string s = str;
    erase_spaces(s);

    // quick validation of valid characters
    if (s.find_first_not_of("0123456789(),") != std::string::npos)
        return StatusCode::SHAPE_WRONG_FORMAT;

    if (s.front() != shapeLeft || s.back() != shapeRight)
        return StatusCode::SHAPE_WRONG_FORMAT;

    s.pop_back();
    s.erase(s.begin());

    auto tokens = tokenize(s, shapeDelimeter);
    shapeInfo.shape.clear();
    std::transform(tokens.begin(), tokens.end(), std::back_inserter(shapeInfo.shape),
        [](const std::string& str) { return std::stoi(str); });

    shapeInfo.shapeMode = FIXED;
    return StatusCode::OK;
}

Status ModelConfig::parseModelMapping() {
    SPDLOG_DEBUG("Parsing model:{} mapping from path:{}", getName(), getPath());
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
        SPDLOG_ERROR("Configuration file is not a valid JSON file.");
        return StatusCode::JSON_INVALID;
    }

    if (validateJsonAgainstSchema(doc, MODELS_MAPPING_INPUTS_SCHEMA) != StatusCode::OK) {
        SPDLOG_WARN("Couldn't load inputs object from file {}", path.c_str());
    } else {
        // Process inputs
        const auto itr = doc.FindMember("inputs");
        for (const auto& key : itr->value.GetObject()) {
            SPDLOG_DEBUG("Loaded input mapping {} => {}", key.name.GetString(), key.value.GetString());
            mappingInputs[key.name.GetString()] = key.value.GetString();
        }
    }

    if (validateJsonAgainstSchema(doc, MODELS_MAPPING_OUTPUTS_SCHEMA) != StatusCode::OK) {
        SPDLOG_WARN("Couldn't load outputs object from file {}", path.c_str());
    } else {
        // Process outputs
        const auto it = doc.FindMember("outputs");
        for (const auto& key : it->value.GetObject()) {
            SPDLOG_DEBUG("Loaded output mapping {} => {}", key.name.GetString(), key.value.GetString());
            mappingOutputs[key.name.GetString()] = key.value.GetString();
        }
    }

    return StatusCode::OK;
}

Status ModelConfig::parseNode(const rapidjson::Value& v) {
    this->setName(v["name"].GetString());
    this->setBasePath(v["base_path"].GetString());

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
            if (!parseShape(shapeInfo, v["shape"].GetString()).ok()) {
                SPDLOG_ERROR("There was an error parsing shape {}", v["shape"].GetString());
            }
            this->addShape(ANONYMOUS_INPUT_NAME, shapeInfo);
        } else {
            if (v["shape"].IsArray()) {
                // Shape for all inputs
                ShapeInfo shapeInfo;
                for (auto& sh : v["shape"].GetArray()) {
                    shapeInfo.shape.push_back(sh.GetUint64());
                }
                this->addShape(ANONYMOUS_INPUT_NAME, shapeInfo);
            } else {
                // Map of shapes
                for (auto& s : v["shape"].GetObject()) {
                    ShapeInfo shapeInfo;
                    // check if legacy format is used
                    if (s.value.IsString()) {
                        if (!ModelConfig::parseShape(shapeInfo, s.value.GetString()).ok()) {
                            SPDLOG_ERROR("There was an error parsing shape {}", v["shape"].GetString());
                        }
                    } else {
                        for (auto& sh : s.value.GetArray()) {
                            shapeInfo.shape.push_back(sh.GetUint64());
                        }
                    }
                    if (s.name.GetString() != ANONYMOUS_INPUT_NAME) {
                        this->addShape(s.name.GetString(), shapeInfo);
                    } else {
                        SPDLOG_WARN("Provided shape name: {} is forbidden and will be omitted", ANONYMOUS_INPUT_NAME);
                    }
                }
            }
        }
    }

    if (v.HasMember("layout")) {
        if (v["layout"].IsString()) {
            this->setLayout(v["layout"].GetString());
        } else {
            for (auto& s : v["layout"].GetObject()) {
                this->addLayout(s.name.GetString(), s.value.GetString());
            }
        }
    }

    if (v.HasMember("plugin_config")) {
        if (!parsePluginConfig(v["plugin_config"]).ok()) {
            SPDLOG_ERROR("Couldn't parse plugin config");
        }
    }

    if (v.HasMember("model_version_policy")) {
        rapidjson::StringBuffer buffer;
        buffer.Clear();
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        v["model_version_policy"].Accept(writer);
        const auto& status = parseModelVersionPolicy(buffer.GetString());
        if (!status.ok()) {
            SPDLOG_ERROR("Couldn't parse model version policy. {}", status.string());
        }
    } else {
        modelVersionPolicy = ModelVersionPolicy::getDefaultVersionPolicy();
    }

    bool batchSizeSet = (getBatchingMode() != FIXED || getBatchSize() != 0);
    bool shapeSet = (getShapes().size() > 0);

    spdlog::debug("Batch size set: {}, shape set: {}", batchSizeSet, shapeSet);
    if (batchSizeSet && shapeSet) {
        spdlog::warn("Both shape and batch size have been defined. Batch size parameter will be ignored.");
        setBatchingMode(FIXED);
        setBatchSize(0);
    }
    return StatusCode::OK;
}

}  // namespace ovms
