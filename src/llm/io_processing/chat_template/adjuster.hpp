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
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ovms {

struct ChatTemplateAdjusterConfig {
    std::vector<std::string> removeFunctions;
    bool concatMultiLineStrings = false;
    std::vector<std::string> removeFunctionsParams;
    bool replaceUndefinedChecks = false;

    static ChatTemplateAdjusterConfig forMinja();
    static ChatTemplateAdjusterConfig forJinja();
};

class BaseChatTemplateAdjustment {
public:
    virtual ~BaseChatTemplateAdjustment() = default;
    virtual void apply(std::string& templateSource) const = 0;
};

class RemoveFunctionParamsAdjustment : public BaseChatTemplateAdjustment {
public:
    explicit RemoveFunctionParamsAdjustment(std::vector<std::string> functions);
    void apply(std::string& templateSource) const override;

private:
    std::vector<std::string> functions;
};

class RemoveFiltersAdjustment : public BaseChatTemplateAdjustment {
public:
    explicit RemoveFiltersAdjustment(std::vector<std::string> functions);
    void apply(std::string& templateSource) const override;

private:
    std::vector<std::string> functions;
};

// TODO @przepeck: It's already implemented in GenAI(but only for pipeline intialization), remove it when it will be supported when setting chat template
class ConcatMultiLineStringAdjustment : public BaseChatTemplateAdjustment {
public:
    ConcatMultiLineStringAdjustment();
    void apply(std::string& templateSource) const override;
};

// TODO @przepeck: It's already implemented in GenAI(but only for pipeline intialization), remove it when it will be supported when setting chat template
class ReplaceUndefinedChecksAdjustment : public BaseChatTemplateAdjustment {
public:
    void apply(std::string& templateSource) const override;
};

class ChatTemplateAdjuster {
public:
    explicit ChatTemplateAdjuster(const ChatTemplateAdjusterConfig& config);

    void adjust(std::string& templateSource) const;

private:
    std::vector<std::unique_ptr<BaseChatTemplateAdjustment>> adjustments;
};

}  // namespace ovms
