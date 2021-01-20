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
#include "custom_node_library_manager.hpp"

#include <utility>

#include <dlfcn.h>

#include "filesystem.hpp"
#include "logging.hpp"

namespace ovms {

Status CustomNodeLibraryManager::loadLibrary(const std::string& name, const std::string& basePath) {
    SPDLOG_LOGGER_INFO(modelmanager_logger, "Loading custom node library name: {}; base_path: {}", name, basePath);

    if (FileSystem::isPathEscaped(basePath)) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Path {} escape with .. is forbidden.", basePath);
        return StatusCode::PATH_INVALID;
    }

    if (libraries.count(name) == 1) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Custom node library name: {} is already loaded", name);
        return StatusCode::NODE_LIBRARY_ALREADY_LOADED;
    }

    void* handle = dlopen(basePath.c_str(), RTLD_LAZY | RTLD_LOCAL);
    char* error = dlerror();
    if (handle == NULL) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Library name: {} failed to open base_path: {} with error: {}", name, basePath, error);
        return StatusCode::NODE_LIBRARY_LOAD_FAILED_OPEN;
    }

    execute_fn execute = reinterpret_cast<execute_fn>(dlsym(handle, "execute"));
    error = dlerror();
    if (error || execute == nullptr) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to load library name: {} with error: {}", name, error);
        dlclose(handle);
        return StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM;
    }

    release_fn releaseBuffer = reinterpret_cast<release_fn>(dlsym(handle, "releaseBuffer"));
    error = dlerror();
    if (error || releaseBuffer == nullptr) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to load library name: {} with error: {}", name, error);
        dlclose(handle);
        return StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM;
    }

    release_fn releaseTensors = reinterpret_cast<release_fn>(dlsym(handle, "releaseTensors"));
    error = dlerror();
    if (error || releaseBuffer == nullptr) {
        SPDLOG_LOGGER_ERROR(modelmanager_logger, "Failed to load library name: {} with error: {}", name, error);
        dlclose(handle);
        return StatusCode::NODE_LIBRARY_LOAD_FAILED_SYM;
    }

    libraries.emplace(std::make_pair(name, NodeLibrary{execute, releaseBuffer, releaseTensors}));

    SPDLOG_LOGGER_INFO(modelmanager_logger, "Successfully loaded custom node library name: {}; base_path: {}", name, basePath);
    return StatusCode::OK;
}

NodeLibrary CustomNodeLibraryManager::getLibrary(const std::string& name) const {
    auto it = libraries.find(name);
    if (it == libraries.end()) {
        return {};
    } else {
        return it->second;
    }
}

}  // namespace ovms
