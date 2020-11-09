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

#include <memory>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

#include <cxxopts.hpp>

#include "customloaderinterface.hpp"

namespace ovms {

Status CustomLoaders::add(std::string name, std::shared_ptr<CustomLoaderInterface> loaderInsterface, void* library) {
    SPDLOG_INFO("Adding loder {} to loaders list",name);
    auto loaderIt = newCustomLoaderInterfacePtrs.find(name);
    if (loaderIt == newCustomLoaderInterfacePtrs.end()) {
        newCustomLoaderInterfacePtrs.insert({name, std::make_pair(library, loaderInsterface)});
        return StatusCode::OK;
    }

    // The loader already exists. return Error   --> Ravikb: Change to appropriate error
    return StatusCode::INTERNAL_ERROR;
}

Status CustomLoaders::remove(std::string& name) {
    SPDLOG_INFO("Removing loder {} from loaders list",name);
    auto loaderIt = customLoaderInterfacePtrs.find(name);

    // The loader is not there. Return error. --> Ravikb: Change to appropriate error
    if (loaderIt == customLoaderInterfacePtrs.end()) {
        return StatusCode::INTERNAL_ERROR;
    }

    customLoaderInterfacePtrs.erase(loaderIt);
    return StatusCode::OK;
}

std::shared_ptr<CustomLoaderInterface> CustomLoaders::find(const std::string& name) {
    SPDLOG_INFO("looking for loder {} in loaders list",name);
    auto loaderIt = customLoaderInterfacePtrs.find(name);

    if (loaderIt == customLoaderInterfacePtrs.end()) {
        return NULL;
    }

    return customLoaderInterfacePtrs[name].second;
}

Status CustomLoaders::move(const std::string& name) {
    SPDLOG_INFO("Moving loder {} from old to new loaders list",name);
    auto loaderIt = customLoaderInterfacePtrs.find(name);

    if (loaderIt == customLoaderInterfacePtrs.end()) {
        return StatusCode::INTERNAL_ERROR;
    }

    newCustomLoaderInterfacePtrs.insert({name, customLoaderInterfacePtrs[name]});
    customLoaderInterfacePtrs.erase(loaderIt);
    return StatusCode::OK;
}

Status CustomLoaders::finalize() {

#if 0
    // By now the remaining loaders in current list are not there in new config. Delete them
    for (auto it = customLoaderInterfacePtrs.begin(); it != customLoaderInterfacePtrs.end(); it++) {
	    SPDLOG_INFO("Loader {} is not there in new list.. deleting the same",it->first);
	    auto loaderPtr = (it->second).second;
	    SPDLOG_INFO("Ravikb::(3) customerLoaderIfPtr refcount = {}",loaderPtr.use_count());
	    customLoaderInterfacePtrs.erase(it);
	    loaderPtr.reset();
	    // TODO--> Ravikb:: Call close function from the library handle
    }
#endif
    SPDLOG_INFO("Clearing the list");
    customLoaderInterfacePtrs.clear();
    
    SPDLOG_INFO("ADding new list to  the old list");
    // now assign new map to servicing map.
    customLoaderInterfacePtrs.insert(newCustomLoaderInterfacePtrs.begin(),newCustomLoaderInterfacePtrs.end());
    newCustomLoaderInterfacePtrs.clear();
    return StatusCode::OK;
}


std::vector<std::string>& CustomLoaders::getNames() {
    currentCustomLoaderNames.clear();

    for (auto it = customLoaderInterfacePtrs.begin(); it != customLoaderInterfacePtrs.end(); it++) {
        currentCustomLoaderNames.emplace_back(it->first);
    }

    return currentCustomLoaderNames;
}

}  // namespace ovms
