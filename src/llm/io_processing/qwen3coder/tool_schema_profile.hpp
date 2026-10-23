//*****************************************************************************
// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
//*****************************************************************************
#pragma once

#include <string>

namespace ovms {
// Qwen XML executable-tool profile. The request handler and parser must agree.
// This is a deliberately restricted Draft 4 profile, not full JSON Schema.
bool isSupportedQwenXmlToolSchema(const std::string& schemaText);
bool validateQwenXmlToolArguments(const std::string& schemaText, const std::string& argumentsText);
}  // namespace ovms
