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
//****************************************************************************

#include "content_parser.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ovms {
namespace {

const std::string MESSAGE_TAG = "<|message|>";
const std::vector<std::string> MESSAGE_END_TAGS = {"<|end|>", "<|return|>", "<|call|>"};
const std::string START_TAG = "<|start|>";
const std::string CHANNEL_TAG = "<|channel|>";

struct TagMatch {
    std::size_t position = std::string::npos;
    std::size_t length = 0;
};

TagMatch findEarliestTag(const std::string& text, const std::vector<std::string>& tags) {
    TagMatch match;
    for (const auto& tag : tags) {
        const std::size_t position = text.find(tag);
        if (position != std::string::npos &&
            (position < match.position || (position == match.position && tag.size() > match.length))) {
            match = {position, tag.size()};
        }
    }
    return match;
}

std::size_t findPartialTagLength(const std::string& text, const std::vector<std::string>& tags) {
    std::size_t partialLength = 0;
    for (const auto& tag : tags) {
        if (tag.empty()) {
            continue;
        }
        const std::size_t maxLength = std::min(text.size(), tag.size() - 1);
        for (std::size_t length = maxLength; length > partialLength; --length) {
            if (text.compare(text.size() - length, length, tag, 0, length) == 0) {
                partialLength = length;
                break;
            }
        }
    }
    return partialLength;
}

}  // namespace

OutputParsingConfig GptOssContentParser::defaultParsingConfig() {
    OutputParsingConfig cfg;
    cfg.startTags = {
        "<|channel|>commentary<|message|>",
        "<|start|>assistant<|channel|>commentary<|message|>",
        "<|channel|>final<|message|>",
        "<|start|>assistant<|channel|>final<|message|>"};
    return cfg;
}

GptOssContentParser::GptOssContentParser(ov::genai::Tokenizer& tokenizer,
    std::optional<OutputParsingConfig> configOverride) :
    BaseOutputParser(tokenizer,
        configOverride.has_value() ? std::move(*configOverride) : defaultParsingConfig()) {}

void GptOssContentParser::resetState() {
    streamState = StreamState::READING_HEADER;
    pendingInput.clear();
    processedChunkSize = 0;
    visibleMessage = false;
}

bool GptOssContentParser::shouldEmitBody(const std::string& header) const {
    if (header.find("to=functions.") != std::string::npos) {
        return false;
    }
    const std::size_t startPos = header.find(START_TAG);
    const std::size_t channelPos = header.rfind(CHANNEL_TAG);
    const std::size_t authorStart = startPos == std::string::npos ? 0 : startPos + START_TAG.size();
    const std::size_t authorEnd = channelPos == std::string::npos ? header.size() : channelPos;
    if (authorEnd < authorStart) {
        return false;
    }
    const std::string author = header.substr(authorStart, authorEnd - authorStart);
    if (author.find("functions.") == 0) {
        return false;
    }

    if (channelPos == std::string::npos) {
        return author.find("assistant") == 0;
    }
    const std::size_t channelStart = channelPos + CHANNEL_TAG.size();
    const std::size_t channelEnd = header.find_first_of(" \t\r\n<", channelStart);
    const std::string channel = header.substr(channelStart,
        channelEnd == std::string::npos ? std::string::npos : channelEnd - channelStart);
    return channel == "final" || channel == "commentary";
}

void GptOssContentParser::appendNewChunk(const std::string& chunk) {
    const std::size_t appendFrom = std::min(processedChunkSize, chunk.size());
    pendingInput.append(chunk, appendFrom, std::string::npos);
    processedChunkSize = chunk.size();
}

GptOssContentParser::ParseProgress GptOssContentParser::consumeHeader(std::string& content) {
    const std::size_t messagePos = pendingInput.find(MESSAGE_TAG);
    if (messagePos == std::string::npos) {
        const std::size_t structuralPos = pendingInput.find("<|");
        if (structuralPos == std::string::npos) {
            content += pendingInput;
            pendingInput.clear();
        } else if (structuralPos > 0) {
            content.append(pendingInput, 0, structuralPos);
            pendingInput.erase(0, structuralPos);
        }
        return ParseProgress::NEED_MORE_INPUT;
    }

    const std::string header = pendingInput.substr(0, messagePos);
    visibleMessage = shouldEmitBody(header);
    pendingInput.erase(0, messagePos + MESSAGE_TAG.size());
    streamState = StreamState::READING_BODY;
    return ParseProgress::ADVANCED;
}

GptOssContentParser::ParseProgress GptOssContentParser::consumeBody(std::string& content, bool& consumedMessage) {
    const TagMatch endTag = findEarliestTag(pendingInput, MESSAGE_END_TAGS);
    if (endTag.position != std::string::npos) {
        const std::string body = pendingInput.substr(0, endTag.position);
        if (visibleMessage && !body.empty()) {
            content += body;
        }
        pendingInput.erase(0, endTag.position + endTag.length);
        consumedMessage = true;
        streamState = StreamState::READING_HEADER;
        visibleMessage = false;
        return ParseProgress::ADVANCED;
    }

    const std::size_t partialLength = findPartialTagLength(pendingInput, MESSAGE_END_TAGS);
    const std::size_t contentLength = pendingInput.size() - partialLength;
    if (visibleMessage && contentLength > 0) {
        content.append(pendingInput, 0, contentLength);
    }
    pendingInput.erase(0, contentLength);
    return ParseProgress::NEED_MORE_INPUT;
}

std::optional<Delta> GptOssContentParser::parseChunk(const std::string& chunk,
    const std::vector<int64_t>& /*tokens*/,
    ov::genai::GenerationFinishReason /*finishReason*/) {
    appendNewChunk(chunk);

    std::string content;
    bool consumedMessage = false;
    while (!pendingInput.empty()) {
        ParseProgress progress;
        switch (streamState) {
        case StreamState::READING_HEADER:
            progress = consumeHeader(content);
            break;
        case StreamState::READING_BODY:
            progress = consumeBody(content, consumedMessage);
            break;
        default:
            throw std::logic_error("Unexpected GPT-OSS content parser state");
        }
        if (progress == ParseProgress::NEED_MORE_INPUT) {
            break;
        }
    }

    if (!content.empty() || consumedMessage) {
        processedChunkSize = 0;
        return ContentDelta{std::move(content)};
    }
    return std::nullopt;
}

}  // namespace ovms
