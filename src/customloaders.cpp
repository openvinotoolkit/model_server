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

#include "customloaders.hpp"

#include "customloaderinterface.hpp"
#include "logging.hpp"
#include "status.hpp"

namespace ovms {

Status CustomLoaders::add(std::string name, std::shared_ptr<CustomLoaderInterface> loaderInterface, void* library) {
    auto loaderIt = newCustomLoaderInterfacePtrs.emplace(name, std::make_pair(library, loaderInterface));
    // if the loader already exists, print an error message
    if (!loaderIt.second) {
        SPDLOG_ERROR("The loader {} already exists in the config file", name);
        return StatusCode::CUSTOM_LOADER_EXISTS;
    }

    return StatusCode::OK;
}

Status CustomLoaders::remove(const std::string& name) {
    SPDLOG_INFO("Removing loader {} from loaders list", name);
    auto loaderIt = customLoaderInterfacePtrs.find(name);

    if (loaderIt == customLoaderInterfacePtrs.end()) {
        return StatusCode::CUSTOM_LOADER_NOT_PRESENT;
    }

    customLoaderInterfacePtrs.erase(loaderIt);
    return StatusCode::OK;
}

std::shared_ptr<CustomLoaderInterface> CustomLoaders::find(const std::string& name) {
    SPDLOG_DEBUG("Looking for loader {} in loaders list", name);
    auto loaderIt = customLoaderInterfacePtrs.find(name);

    if (loaderIt == customLoaderInterfacePtrs.end()) {
        return nullptr;
    }

    return (loaderIt->second).second;
}

Status CustomLoaders::move(const std::string& name) {
    SPDLOG_INFO("Moving loader {} from old to new loaders list", name);
    auto loaderIt = customLoaderInterfacePtrs.find(name);

    if (loaderIt == customLoaderInterfacePtrs.end()) {
        return StatusCode::INTERNAL_ERROR;
    }

    newCustomLoaderInterfacePtrs.insert({name, customLoaderInterfacePtrs[name]});
    customLoaderInterfacePtrs.erase(loaderIt);
    return StatusCode::OK;
}

Status CustomLoaders::finalize() {
    // By now the remaining loaders in current list are not there in new config. Delete them
    for (auto it = customLoaderInterfacePtrs.begin(); it != customLoaderInterfacePtrs.end(); it++) {
        SPDLOG_INFO("Loader {} is not there in new list.. deleting the same", it->first);
        auto loaderPtr = (it->second).second;
        loaderPtr->loaderDeInit();
    }

    SPDLOG_INFO("Clearing the list");
    customLoaderInterfacePtrs.clear();

    SPDLOG_INFO("Adding new list to the old list");
    // now assign new map to servicing map.
    customLoaderInterfacePtrs.insert(newCustomLoaderInterfacePtrs.begin(), newCustomLoaderInterfacePtrs.end());
    newCustomLoaderInterfacePtrs.clear();
    return StatusCode::OK;
}

}  // namespace ovms
