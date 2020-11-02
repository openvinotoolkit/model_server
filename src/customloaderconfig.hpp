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
#pragma once

#include <string>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <spdlog/spdlog.h>

#include "status.hpp"
#include "stringutils.hpp"

namespace ovms {

/**
     * @brief This class represents Custom Loader configuration
     */
class CustomLoaderConfig {
private:
    /**
         * @brief Custom Loader Name
         */
    std::string loaderName;

    /**
         * @brief Custom Loader Library Path
         */
    std::string libraryPath;

    /**
         * @brief Custom Loader Config Path
         */
    std::string loaderConfigFile;

public:
    /**
         * @brief Construct a new Custom Loader Config object
         *
         * @param name
         * @param libraryPath
         * @param configPath
         */
    CustomLoaderConfig(const std::string& loaderName = "",
        const std::string& libraryPath = "",
        const std::string& loaderConfigFile = "") :
        loaderName(loaderName),
        libraryPath(libraryPath),
        loaderConfigFile(loaderConfigFile) {
    }

    void clear() {
        loaderName.clear();
        libraryPath.clear();
        loaderConfigFile.clear();
    }

    /**
         * @brief Get the name
         *
         * @return const std::string&
         */
    const std::string& getLoaderName() const {
        return this->loaderName;
    }

    /**
         * @brief Set the name
         *
         * @param name
         */
    void setLoaderName(const std::string& loaderName) {
        this->loaderName = loaderName;
    }

    /**
         * @brief Get the Library Path
         *
         * @return const std::string&
         */
    const std::string& getLibraryPath() const {
        return this->libraryPath;
    }

    /**
         * @brief Set the Library Path
         *
         * @param libraryPath
         */
    void setLibraryPath(const std::string& libraryPath) {
        this->libraryPath = libraryPath;
    }

    /**
         * @brief Get the Config Path
         *
         * @return const std::string&
         */
    const std::string& getLoaderConfigFile() const {
        return this->loaderConfigFile;
    }

    /**
         * @brief Set the Config Path
         *
         * @param configPath
         */
    void setLoaderConfigFile(const std::string& loaderConfigFile) {
        this->loaderConfigFile = loaderConfigFile;
    }

    /**
     * @brief  Parses all settings from a JSON node
        *
        * @return Status
        */
    Status parseNode(const rapidjson::Value& v) {
        try {
            this->setLoaderName(v["loader_name"].GetString());
            this->setLibraryPath(v["library_path"].GetString());
            if (v.HasMember("config_path"))
                this->setLoaderConfigFile(v["loader_config_file"].GetString());
        } catch (...) {
            spdlog::error("There was an error parsing the custom loader config");
            return StatusCode::JSON_INVALID;
        }
        return StatusCode::OK;
    }
};
}  // namespace ovms
