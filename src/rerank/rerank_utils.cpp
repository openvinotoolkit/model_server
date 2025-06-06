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

#include "rerank_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#pragma warning(push)
#pragma warning(disable : 6313)
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include "../logging.hpp"
#include "../profiler.hpp"

using namespace rapidjson;

namespace ovms {

absl::Status RerankHandler::parseRequest() {
    // Parsed JSON is not guaranteed to be valid, we may reach this point via multipart content type request with no valid JSON parser
    if (doc.HasParseError()) {
        return absl::InvalidArgumentError("Non-json request received in rerank calculator");
    }

    // model: string
    auto it = doc.FindMember("model");
    if (it != doc.MemberEnd()) {
        if (!it->value.IsString())
            return absl::InvalidArgumentError("model accepts string values");
        request.model = it->value.GetString();
    }
    // query: string
    it = doc.FindMember("query");
    if (it == doc.MemberEnd()) {
        return absl::InvalidArgumentError("query field is missing in request");
    }
    if (!it->value.IsString()) {
        return absl::InvalidArgumentError("query is not a string");
    } else {
        request.query = it->value.GetString();
    }
    // documents: list or map of strings
    it = doc.FindMember("documents");
    if (it == doc.MemberEnd()) {
        return absl::InvalidArgumentError("documents field is missing in request");
    }
    if (!it->value.IsArray()) {
        return absl::InvalidArgumentError("documents is not an array");
    } else {
        for (auto& d : it->value.GetArray()) {
            if (!d.IsString() && !d.IsObject()) {
                return absl::InvalidArgumentError("documents array element is neither string nor object");
            }
            if (d.IsString()) {
                if (request.documentsMap.size() > 0) {
                    return absl::InvalidArgumentError("all documents have to be the same type (string or objects)");
                }
                request.documentsList.push_back(d.GetString());
            } else if (d.IsObject()) {
                if (request.documentsList.size() > 0) {
                    return absl::InvalidArgumentError("all documents have to be the same type (string or objects)");
                }
                if (!d.GetObject().HasMember("title")) {
                    return absl::InvalidArgumentError("document title field is missing");
                }
                if (!d.GetObject()["title"].IsString()) {
                    return absl::InvalidArgumentError("document title field have to be string");
                }
                if (!d.GetObject().HasMember("text")) {
                    return absl::InvalidArgumentError("document text field is missing");
                }
                if (!d.GetObject()["text"].IsString()) {
                    return absl::InvalidArgumentError("document text field have to be string");
                }
                request.documentsMap.insert({d.GetObject()["title"].GetString(), d.GetObject()["text"].GetString()});
            }
        }
    }
    // top_n: int; optional
    it = doc.FindMember("top_n");
    if ((it != doc.MemberEnd()) && (!it->value.IsNull())) {
        if (!it->value.IsInt())
            return absl::InvalidArgumentError("top_n accepts integer values");
        request.topN = it->value.GetInt();
    } else {
        if (request.documentsList.size() > 0) {
            request.topN = request.documentsList.size();
        } else {
            request.topN = request.documentsMap.size();
        }
    }
    // rank_fields: list of strings; optional
    it = doc.FindMember("rank_fields");
    if ((it != doc.MemberEnd()) && !(it->value.IsNull())) {
        if (!it->value.IsArray()) {
            return absl::InvalidArgumentError("rank_fields is not an array");
        } else {
            request.rankFields = std::vector<std::string>();
            for (auto& d : it->value.GetArray()) {
                if (!d.IsString()) {
                    return absl::InvalidArgumentError("rank_fields array element is not a string");
                }
                request.rankFields.value().push_back(d.GetString());
            }
        }
    }
    // return_documents: boolean; optional
    it = doc.FindMember("return_documents");
    if ((it != doc.MemberEnd()) && !(it->value.IsNull())) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("return_documents accepts boolean values");
        request.returnDocuments = it->value.GetBool();
    }

    // max_chunks_per_doc: int; optional
    it = doc.FindMember("max_chunks_per_doc");
    if ((it != doc.MemberEnd()) && !(it->value.IsNull())) {
        if (!it->value.IsInt())
            return absl::InvalidArgumentError("max_chunks_per_doc accepts integer values");
        request.maxChunksPerDoc = it->value.GetInt();
    }

    return absl::OkStatus();
}

std::vector<size_t> getSortedIndexes(const std::vector<float>& scores) {
    std::vector<size_t> indexes(scores.size());
    std::iota(indexes.begin(), indexes.end(), 0);

    std::sort(indexes.begin(), indexes.end(),
        [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

    return indexes;
}

absl::Status RerankHandler::parseResponse(StringBuffer& buffer, std::vector<float>& scores) {
    Writer<StringBuffer> writer(buffer);
    writer.StartObject();

    writer.String("results");
    writer.StartArray();
    auto sortedIndexes = getSortedIndexes(scores);
    for (size_t i = 0; i < sortedIndexes.size(); i++) {
        if (i >= request.topN) {
            break;
        }
        auto index = sortedIndexes[i];
        writer.StartObject();
        writer.String("index");
        writer.Int(index);
        writer.String("relevance_score");
        writer.Double(scores[index]);
        if (request.returnDocuments.has_value() && request.returnDocuments.value()) {
            if (request.documentsList.size() > index) {
                writer.String("document");
                writer.StartObject();
                writer.String("text");
                writer.String(request.documentsList[index].c_str());
                writer.EndObject();
            } else {
                return absl::InvalidArgumentError("document map not supported");  // TODO add support for document map
            }
        }
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();

    return absl::OkStatus();
}

std::string RerankHandler::getModel() const { return request.model; }
std::string RerankHandler::getQuery() const { return request.query; }
const std::vector<std::string>& RerankHandler::getDocumentsList() const { return request.documentsList; }
const std::unordered_map<std::string, std::string>& RerankHandler::getDocumentsMap() const { return request.documentsMap; }
std::optional<int> RerankHandler::getTopN() const { return request.topN; }
std::optional<bool> RerankHandler::getReturnDocuments() const { return request.returnDocuments; }
std::optional<std::vector<std::string>> RerankHandler::getRankFields() const { return request.rankFields; }
std::optional<int> RerankHandler::getMaxChunksPerDoc() const { return request.maxChunksPerDoc; }

// Takes tokenizer outputs: input_ids and attention_mask and chunks them into batches of smaller width (max_tokens_per_chunk param).
// Preserves the chunk-original_document mapping for later use in chunk_mapping variable.
// If max_tokens_per_chunk is bigger than the longest document, no chunking is needed.
absl::Status chunkDocuments(
    const ov::Tensor& in_input_ids, const ov::Tensor& in_attention_mask,
    ov::Tensor& out_input_ids, ov::Tensor& out_attention_mask,
    std::vector<size_t>& chunk_mapping, size_t max_tokens_per_chunk,
    size_t max_allowed_chunks, int64_t pad_token) {

    if (max_tokens_per_chunk == 0) {
        return absl::InvalidArgumentError("no space left for chunks");
    }

    if (in_input_ids.get_shape() != in_attention_mask.get_shape()) {
        return absl::InvalidArgumentError("input_ids and attention_mask shapes do not match");
    }

    if (in_input_ids.get_shape().size() != 2) {
        return absl::InvalidArgumentError("input_ids and attention_mask should be 2D tensors");
    }

    if (in_input_ids.get_element_type() != ov::element::i64) {
        return absl::InvalidArgumentError("input_ids and attention_mask should be int64 tensors");
    }

    if (in_input_ids.get_element_type() != in_attention_mask.get_element_type()) {
        return absl::InvalidArgumentError("input_ids and attention_mask should have the same element type");
    }

    size_t tokens_count_of_longest_document = in_input_ids.get_shape()[1];
    size_t batch_size = in_input_ids.get_shape()[0];
    if (batch_size > max_allowed_chunks) {
        return absl::InvalidArgumentError(std::string{"exceeding max_allowed_chunks before chunking limit: "} + std::to_string(max_allowed_chunks) + std::string{"; actual: "} + std::to_string(batch_size));
    }

    if (tokens_count_of_longest_document <= max_tokens_per_chunk) {
        out_input_ids = in_input_ids;
        out_attention_mask = in_attention_mask;
        chunk_mapping.resize(batch_size);
        std::iota(chunk_mapping.begin(), chunk_mapping.end(), 0);
        return absl::OkStatus();
    }

    size_t new_tokens_count_of_longest_document = 0;
    for (size_t i = 0; i < batch_size; i++) {
        int64_t* in_attention_mask_data = reinterpret_cast<int64_t*>(in_attention_mask.data()) + i * tokens_count_of_longest_document;
        auto it = std::find(in_attention_mask_data, in_attention_mask_data + tokens_count_of_longest_document, 0);
        size_t token_count = (it != in_attention_mask_data + tokens_count_of_longest_document) ? std::distance(in_attention_mask_data, it) : tokens_count_of_longest_document;
        if (token_count > max_tokens_per_chunk) {
            size_t number_of_new_chunks = (token_count + max_tokens_per_chunk - 1) / max_tokens_per_chunk;
            for (size_t j = 0; j < number_of_new_chunks; j++) {
                chunk_mapping.push_back(i);
            }
            new_tokens_count_of_longest_document = std::max(max_tokens_per_chunk, new_tokens_count_of_longest_document);
        } else {
            chunk_mapping.push_back(i);
            new_tokens_count_of_longest_document = std::max(token_count, new_tokens_count_of_longest_document);
        }
    }

    size_t new_batch_size = chunk_mapping.size();
    if (new_batch_size > max_allowed_chunks) {
        return absl::InvalidArgumentError(std::string{"exceeding max_allowed_chunks after chunking limit: "} + std::to_string(max_allowed_chunks) + std::string{"; actual: "} + std::to_string(new_batch_size));
    }

    if (new_batch_size != batch_size) {
        SPDLOG_LOGGER_DEBUG(rerank_calculator_logger, "Chunking required, initial batch size: {}, final batch size: {}", batch_size, new_batch_size);
    }

    out_input_ids = ov::Tensor(ov::element::i64, ov::Shape{new_batch_size, new_tokens_count_of_longest_document});
    out_attention_mask = ov::Tensor(ov::element::i64, ov::Shape{new_batch_size, new_tokens_count_of_longest_document});

    size_t new_i = 0;
    for (size_t i = 0; i < batch_size; i++) {
        int64_t* in_input_ids_data = reinterpret_cast<int64_t*>(in_input_ids.data()) + i * tokens_count_of_longest_document;
        int64_t* in_attention_mask_data = reinterpret_cast<int64_t*>(in_attention_mask.data()) + i * tokens_count_of_longest_document;
        auto it = std::find(in_attention_mask_data, in_attention_mask_data + tokens_count_of_longest_document, 0);
        size_t token_count = (it != in_attention_mask_data + tokens_count_of_longest_document) ? std::distance(in_attention_mask_data, it) : tokens_count_of_longest_document;
        if (token_count > max_tokens_per_chunk) {
            size_t number_of_new_chunks = (token_count + max_tokens_per_chunk - 1) / max_tokens_per_chunk;
            for (size_t j = 0; j < number_of_new_chunks; j++) {
                size_t start = j * max_tokens_per_chunk;
                size_t end = std::min(start + max_tokens_per_chunk, token_count);
                size_t new_chunk_size = end - start;

                int64_t* new_doc_input_ids_data = reinterpret_cast<int64_t*>(out_input_ids.data()) + new_i * new_tokens_count_of_longest_document;
                int64_t* new_doc_attention_mask_data = reinterpret_cast<int64_t*>(out_attention_mask.data()) + new_i * new_tokens_count_of_longest_document;

                std::fill(new_doc_input_ids_data, new_doc_input_ids_data + new_tokens_count_of_longest_document, pad_token);
                std::copy(in_input_ids_data + start, in_input_ids_data + start + new_chunk_size, new_doc_input_ids_data);
                std::fill(new_doc_attention_mask_data, new_doc_attention_mask_data + new_tokens_count_of_longest_document, 0);
                std::copy(in_attention_mask_data + start, in_attention_mask_data + start + new_chunk_size, new_doc_attention_mask_data);
                new_i++;
            }
        } else {
            int64_t* new_doc_input_ids_data = reinterpret_cast<int64_t*>(out_input_ids.data()) + new_i * new_tokens_count_of_longest_document;
            int64_t* new_doc_attention_mask_data = reinterpret_cast<int64_t*>(out_attention_mask.data()) + new_i * new_tokens_count_of_longest_document;

            std::fill(new_doc_input_ids_data, new_doc_input_ids_data + new_tokens_count_of_longest_document, pad_token);
            std::copy(in_input_ids_data, in_input_ids_data + token_count, new_doc_input_ids_data);
            std::fill(new_doc_attention_mask_data, new_doc_attention_mask_data + new_tokens_count_of_longest_document, 0);
            std::copy(in_attention_mask_data, in_attention_mask_data + token_count, new_doc_attention_mask_data);
            new_i++;
        }
    }

    return absl::OkStatus();
}

}  // namespace ovms
