//*****************************************************************************
// Copyright 2026 Intel Corporation
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
#include "device_display_owner_module.hpp"

#include <fstream>
#include <string>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <va/va.h>
#include <va/va_drm.h>
#endif

#include "config.hpp"
#include "logging.hpp"
#include "modelinstance.hpp"
#include "mpi/intel_mpi.h"
#include "status.hpp"

namespace ovms {
namespace {

Status parseVaDisplayConfig(const Config& config, std::string& targetDevice, std::string& renderNode) {
    if (config.configPath().empty()) {
        return StatusCode::OK;
    }

    std::ifstream configFile(config.configPath());
    if (!configFile.good()) {
        return Status(StatusCode::JSON_INVALID, "Cannot open config file for va_display");
    }
    rapidjson::IStreamWrapper stream(configFile);
    rapidjson::Document document;
    if (document.ParseStream(stream).HasParseError()) {
        return Status(StatusCode::JSON_INVALID, "Cannot parse config file for va_display");
    }
    const auto vaDisplayMember = document.FindMember("va_display");
    if (vaDisplayMember == document.MemberEnd()) {
        return StatusCode::OK;
    }
    if (!vaDisplayMember->value.IsObject()) {
        return Status(StatusCode::JSON_INVALID, "va_display must be an object with target_device and render_node");
    }
    const auto targetDeviceMember = vaDisplayMember->value.FindMember("target_device");
    const auto renderNodeMember = vaDisplayMember->value.FindMember("render_node");
    if (targetDeviceMember == vaDisplayMember->value.MemberEnd() || !targetDeviceMember->value.IsString() ||
        renderNodeMember == vaDisplayMember->value.MemberEnd() || !renderNodeMember->value.IsString()) {
        return Status(StatusCode::JSON_INVALID, "va_display requires string target_device and render_node");
    }
    targetDevice = targetDeviceMember->value.GetString();
    renderNode = renderNodeMember->value.GetString();
    if (targetDevice.rfind("GPU", 0) != 0 || renderNode.empty()) {
        return Status(StatusCode::JSON_INVALID, "va_display target_device must be a GPU device and render_node must not be empty");
    }
    return StatusCode::OK;
}

}  // namespace

Status DeviceDisplayOwnerModule::start(const Config& config) {
    std::string targetDevice;
    std::string renderNode;
    Status status = parseVaDisplayConfig(config, targetDevice, renderNode);
    if (!status.ok()) {
        return status;
    }
    if (renderNode.empty()) {
        state = ModuleState::INITIALIZED;
        return StatusCode::OK;
    }
#ifdef __linux__
    renderNodeFd = open(renderNode.c_str(), O_RDWR);
    if (renderNodeFd < 0) {
        return Status(StatusCode::INTERNAL_ERROR, "Cannot open configured va_display render_node: " + renderNode);
    }
    vaDisplay = vaGetDisplayDRM(renderNodeFd);
    if (vaDisplay == nullptr) {
        close(renderNodeFd);
        renderNodeFd = -1;
        return Status(StatusCode::INTERNAL_ERROR, "Cannot create VA display for configured render_node: " + renderNode);
    }
    int major = 0;
    int minor = 0;
    if (vaInitialize(vaDisplay, &major, &minor) != VA_STATUS_SUCCESS) {
        vaDisplay = nullptr;
        close(renderNodeFd);
        renderNodeFd = -1;
        return Status(StatusCode::INTERNAL_ERROR, "Cannot initialize VA display for configured render_node: " + renderNode);
    }
    // TODO: replace this server-level owner with per-device ownership when
    // globalVaDisplay becomes per-device for multi-GPU support.
    setGlobalVaDisplay(vaDisplay, targetDevice);
    imp_video_set_va_display(vaDisplay);
    SPDLOG_INFO("VA display owner started for target device {} on {} (VA-API {}.{})", targetDevice, renderNode, major, minor);
    state = ModuleState::INITIALIZED;
    return StatusCode::OK;
#else
    return Status(StatusCode::NOT_IMPLEMENTED, "va_display is only supported on Linux");
#endif
}

void DeviceDisplayOwnerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN) {
        return;
    }
#ifdef __linux__
    imp_video_set_va_display(nullptr);
    setGlobalVaDisplay(nullptr);
    if (vaDisplay != nullptr) {
        vaTerminate(vaDisplay);
        vaDisplay = nullptr;
    }
    if (renderNodeFd >= 0) {
        close(renderNodeFd);
        renderNodeFd = -1;
    }
#endif
    state = ModuleState::SHUTDOWN;
}

DeviceDisplayOwnerModule::~DeviceDisplayOwnerModule() {
    shutdown();
}

}  // namespace ovms