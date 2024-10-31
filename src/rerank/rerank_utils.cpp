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

#include <rapidjson/document.h>
#include <rapidjson/writer.h>

#include "../logging.hpp"
#include "../profiler.hpp"

using namespace rapidjson;

namespace ovms {

absl::Status RerankHandler::parseRequest() {
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
    if (it != doc.MemberEnd()) {
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
    if (it != doc.MemberEnd()) {
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
    if (it != doc.MemberEnd()) {
        if (!it->value.IsBool())
            return absl::InvalidArgumentError("return_documents accepts boolean values");
        request.returnDocuments = it->value.GetBool();
    }

    // max_chunks_per_doc: int; optional
    it = doc.FindMember("max_chunks_per_doc");
    if (it != doc.MemberEnd()) {
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
        if(i >= request.topN){
            break;
        }
        auto index = sortedIndexes[i];
        writer.StartObject();
        writer.String("index");
        writer.Int(index);
        writer.String("relevance_score");
        writer.Double(scores[index]);
        if (request.returnDocuments) {
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
std::vector<std::string> RerankHandler::getDocumentsList() const { return request.documentsList; }
std::unordered_map<std::string, std::string> RerankHandler::getDocumentsMap() const { return request.documentsMap; }
std::optional<int> RerankHandler::getTopN() const { return request.topN; }
std::optional<bool> RerankHandler::getReturnDocuments() const { return request.returnDocuments; }
std::optional<std::vector<std::string>> RerankHandler::getRankFields() const { return request.rankFields; }
std::optional<int> RerankHandler::getMaxChunksPerDoc() const { return request.maxChunksPerDoc; }

}  // namespace ovms
