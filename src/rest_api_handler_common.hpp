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

#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>


#include "status.hpp"

namespace tensorflow::serving::net_http {
class ServerRequestInterface;
}

namespace ovms {

enum RequestType { Predict,
    GetModelStatus,
    GetModelMetadata,
    ConfigReload,
    ConfigStatus,
    KFS_GetModelReady,
    KFS_Infer,
    KFS_GetModelMetadata,
    KFS_GetServerReady,
    KFS_GetServerLive,
    KFS_GetServerMetadata,
    V3,
    Metrics };

struct HttpResponseComponents {
    std::optional<int> inferenceHeaderContentLength;
};

struct HttpRequestComponents {
    RequestType type;
    std::string_view http_method;
    std::string model_name;
    std::optional<int64_t> model_version;
    std::optional<std::string_view> model_version_label;
    std::string processing_method;
    std::string model_subresource;
    std::optional<int> inferenceHeaderContentLength;
    std::vector<std::pair<std::string, std::string>> headers;
};

using HandlerCallbackFn = std::function<Status(const std::string_view, const HttpRequestComponents&, std::string&, const std::string&, HttpResponseComponents&, tensorflow::serving::net_http::ServerRequestInterface*)>;

}  // namespace ovms
