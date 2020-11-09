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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "customloaderinterface.hpp"
#include "status.hpp"

namespace ovms {
/**
     * @brief Provides all customloaders
     */
class CustomLoaders {
private:
    /**
         * @brief A default constructor is private
         */
    CustomLoaders() = default;

    /**
         * @brief Private copying constructor
         */
    CustomLoaders(const CustomLoaders&) = delete;

    std::map<std::string, std::pair<void*, std::shared_ptr<CustomLoaderInterface>>> customLoaderInterfacePtrs;
    std::map<std::string, std::pair<void*, std::shared_ptr<CustomLoaderInterface>>> newCustomLoaderInterfacePtrs;

    std::vector<std::string> currentCustomLoaderNames;

public:
    /**
         * @brief Gets the instance of the CustomLoaders
         */
    static CustomLoaders& instance() {
        static CustomLoaders instance;
        return instance;
    }

    /**
         * @brief insert a new customloader
         * 
         * @return status 
         */
    Status add(std::string name, std::shared_ptr<CustomLoaderInterface> loaderInsterface, void* library);

    /**
         * @brief remove an existing customLoader referenced by it's name
         * 
         * @return status 
         */
    Status remove(std::string& name);

    /**
         * @brief find an existing customLoader referenced by it's name
         * 
         * @return pointer to customloader Interface if found, else NULL
         */
    std::shared_ptr<CustomLoaderInterface> find(const std::string& name);

    /**
         * @brief move the existing loader from serviced map to new map.
         * 
         * @return status
         */
    Status move(const std::string& name);

    /**
         * @brief finalize the customloaders list after parsing config
         * 
         * @return status
         */
    Status finalize();

    /**
         * @brief build a list of all customloader names
         * 
         * @return vector of string
         */
    std::vector<std::string>& getNames();
};
}  // namespace ovms
