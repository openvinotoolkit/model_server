//*****************************************************************************
// Copyright 2025 Intel Corporation
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
#include <iostream>
#include <map>
#include <string>
#pragma once
namespace ovms {

enum GraphExportType : unsigned int {
    TEXT_GENERATION_GRAPH,
    RERANK_GRAPH,
    EMBEDDINGS_GRAPH,
    IMAGE_GENERATION_GRAPH,
    UNKNOWN_GRAPH
};

const std::map<GraphExportType, std::string> typeToString = {
    {TEXT_GENERATION_GRAPH, "text_generation"},
    {RERANK_GRAPH, "rerank"},
    {EMBEDDINGS_GRAPH, "embeddings"},
    {IMAGE_GENERATION_GRAPH, "image_generation"},
    {UNKNOWN_GRAPH, "unknown_graph"}};

const std::map<std::string, GraphExportType> stringToType = {
    {"text_generation", TEXT_GENERATION_GRAPH},
    {"rerank", RERANK_GRAPH},
    {"embeddings", EMBEDDINGS_GRAPH},
    {"image_generation", IMAGE_GENERATION_GRAPH},
    {"unknown_graph", UNKNOWN_GRAPH}};

std::string enumToString(GraphExportType type);
GraphExportType stringToEnum(std::string inString);

}  // namespace ovms
