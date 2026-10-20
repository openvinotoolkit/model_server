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

#include "port/rapidjson_document.hpp"
#include "port/rapidjson_istreamwrapper.hpp"

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

#ifdef _WIN32
// Included last: windows.h defines macros (e.g. interface) that corrupt the
// OVMS headers above if included before them.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d11.h>
#endif

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

#ifdef _WIN32
// Windows D3D11 analog of parseVaDisplayConfig. An optional "d3d11_device" config
// object enables server-owned D3D11 device ownership; its "target_device" (default
// "GPU") must be a GPU device. Absence leaves Windows D3D11 ownership disabled.
Status parseD3D11DeviceConfig(const Config& config, bool& enabled, std::string& targetDevice) {
    enabled = false;
    targetDevice = "GPU";
    if (config.configPath().empty()) {
        return StatusCode::OK;
    }
    std::ifstream configFile(config.configPath());
    if (!configFile.good()) {
        return Status(StatusCode::JSON_INVALID, "Cannot open config file for d3d11_device");
    }
    rapidjson::IStreamWrapper stream(configFile);
    rapidjson::Document document;
    if (document.ParseStream(stream).HasParseError()) {
        return Status(StatusCode::JSON_INVALID, "Cannot parse config file for d3d11_device");
    }
    const auto member = document.FindMember("d3d11_device");
    if (member == document.MemberEnd()) {
        return StatusCode::OK;
    }
    if (!member->value.IsObject()) {
        return Status(StatusCode::JSON_INVALID, "d3d11_device must be an object");
    }
    const auto targetDeviceMember = member->value.FindMember("target_device");
    if (targetDeviceMember != member->value.MemberEnd()) {
        if (!targetDeviceMember->value.IsString()) {
            return Status(StatusCode::JSON_INVALID, "d3d11_device target_device must be a string");
        }
        targetDevice = targetDeviceMember->value.GetString();
        if (targetDevice.rfind("GPU", 0) != 0) {
            return Status(StatusCode::JSON_INVALID, "d3d11_device target_device must be a GPU device");
        }
    }
    enabled = true;
    return StatusCode::OK;
}
#endif

}  // namespace

Status DeviceDisplayOwnerModule::start(const Config& config) {
#ifdef _WIN32
    bool d3d11Enabled = false;
    std::string d3d11TargetDevice;
    Status d3d11Status = parseD3D11DeviceConfig(config, d3d11Enabled, d3d11TargetDevice);
    if (!d3d11Status.ok()) {
        return d3d11Status;
    }
    if (!d3d11Enabled) {
        state = ModuleState::INITIALIZED;
        return StatusCode::OK;
    }
    // Create one server-owned ID3D11Device (video-capable) before GPU models
    // load, so ModelManager compiles them on a matching D3DContext and GStreamer
    // decodes onto the same device. The device outlives models and graphs; it is
    // released in shutdown() after the servable manager has stopped.
    ID3D11Device* device = nullptr;
    D3D_FEATURE_LEVEL featureLevel{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   D3D11_CREATE_DEVICE_VIDEO_SUPPORT, nullptr, 0,
                                   D3D11_SDK_VERSION, &device, &featureLevel, nullptr);
    if (FAILED(hr) || device == nullptr) {
        return Status(StatusCode::INTERNAL_ERROR, "Cannot create server-owned D3D11 device for d3d11_device");
    }
    d3d11Device = device;
    setGlobalD3D11Device(device);
    imp_video_set_d3d11_device(device);
    SPDLOG_INFO("D3D11 device owner started for target device {}", d3d11TargetDevice);
    state = ModuleState::INITIALIZED;
    return StatusCode::OK;
#else
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
#endif
}

void DeviceDisplayOwnerModule::shutdown() {
    if (state == ModuleState::SHUTDOWN) {
        return;
    }
#ifdef _WIN32
    imp_video_set_d3d11_device(nullptr);
    setGlobalD3D11Device(nullptr);
    if (d3d11Device != nullptr) {
        reinterpret_cast<ID3D11Device*>(d3d11Device)->Release();
        d3d11Device = nullptr;
    }
#endif
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