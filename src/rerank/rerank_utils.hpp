//*****************************************************************************
// Copyright 2024 Intel Corporation
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

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>

#include "absl/status/status.h"

using namespace rapidjson;

namespace ovms {

// Class that maps Rerank request content.
struct RerankRequest {
    std::string model;
    std::string query;
    std::vector<std::string> documentsList;
    std::unordered_map<std::string, std::string> documentsMap;
    std::optional<int> topN{std::nullopt};
    std::optional<std::vector<std::string>> rankFields{std::nullopt};
    std::optional<bool> returnDocuments{std::nullopt};
    std::optional<int> maxChunksPerDoc{std::nullopt};

    RerankRequest() = default;
    ~RerankRequest() = default;
};

// Class that wraps rerank request, holds and processes raw JSON, provides methods for serialization and keeps track of usage.
// It is used in the calculator.
class RerankHandler {
    Document& doc;
    RerankRequest request;

public:
    RerankHandler(Document& doc) :
        doc(doc) {}

    std::string getModel() const;
    std::string getQuery() const;
    const std::vector<std::string>& getDocumentsList() const;
    const std::unordered_map<std::string, std::string>& getDocumentsMap() const;

    std::optional<int> getTopN() const;
    std::optional<bool> getReturnDocuments() const;
    std::optional<std::vector<std::string>> getRankFields() const;
    std::optional<int> getMaxChunksPerDoc() const;

    absl::Status parseRequest();
    absl::Status parseResponse(StringBuffer& buffer, std::vector<float>& scores);
};

// Takes tokenizer outputs: input_ids and attention_mask and chunks them into batches of smaller width (max_tokens_per_chunk param).
// Preserves the chunk-original_document mapping for later use in chunk_mapping variable.
// If max_tokens_per_chunk is bigger than the longest document, no chunking is needed.
absl::Status chunkDocuments(
    const ov::Tensor& in_input_ids, const ov::Tensor& in_attention_mask,
    ov::Tensor& out_input_ids, ov::Tensor& out_attention_mask,
    std::vector<size_t>& chunk_mapping, size_t max_tokens_per_chunk,
    size_t max_allowed_chunks, int64_t pad_token);

}  // namespace ovms
