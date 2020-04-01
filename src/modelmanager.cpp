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
#include <fstream>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include "modelmanager.h"

namespace ovms {

Status ModelManager::start(const std::string& jsonFilename) {
    rapidjson::Document doc;
    std::ifstream ifs(jsonFilename.c_str());

    // Perform some basic checks on the config file
    if (!ifs.good()) {
        // Logger(Log::Error, "File is invalid ", jsonFilename);
        return Status::FILE_INVALID;
    }

    rapidjson::IStreamWrapper isw(ifs);
    if (doc.ParseStream(isw).HasParseError()) {
        // Logger(Log::Error, "Configuration file is not a valid JSON file.");
        return Status::JSON_INVALID;
    }

    // TODO validate json against schema
    const auto itr = doc.FindMember("models");
    if (itr == doc.MemberEnd() || !itr->value.IsArray()) {
        // Logger(Log::Error, "Configuration file doesn't have models property.");
        return Status::JSON_INVALID;
    }
    
    for (auto& v : itr->value.GetArray()) {
        Model model;
        std::string name = v["name"].GetString();
        if (models.find(name) == models.end()) {
            models[name] = model;
        }

        std::vector<size_t> shape;
        for (auto& s : v["shape"].GetArray()) {
            shape.push_back(s.GetUint64());
        }

        auto status = models[name].addVersion(v["name"].GetString(),
                                              v["path"].GetString(),
                                              v["backend"].GetString(),
                                              v["version"].GetInt64(),
                                              v["batchSize"].GetUint64(),
                                              shape);
        if (status != Status::OK) {
            // Logger(Log::Warning, "There was an error loading a model ", v["name"].GetString());
        }
    }

    return Status::OK;
}

Status ModelManager::join() {
    if (monitor.joinable()) {
        monitor.join();
    }

    return Status::OK;
}
  
} // namespace ovms