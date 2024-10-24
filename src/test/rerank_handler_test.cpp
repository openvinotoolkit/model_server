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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../rerank/rerank_utils.hpp"

using namespace ovms;

using ::testing::ElementsAre;

class RerankHandlerTest : public ::testing::Test {
protected:
    Document doc;
    std::string json;
    void SetUp() override {
    }
};

TEST_F(RerankHandlerTest, ValidRequestDocumentsMap) {
    json = R"({
    "model": "model",
    "query": "query",
    "documents": [
        {
        "title": "first document title",
        "text": "first document text"
        },
        {
        "title": "second document title",
        "text": "second document text"
        }
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    EXPECT_STREQ(handler.getModel().c_str(), "model");
    EXPECT_STREQ(handler.getQuery().c_str(), "query");
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 2);
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getDocumentsList().size(), 0);
    ASSERT_EQ(handler.getDocumentsMap().size(), 2);
    std::unordered_map<std::string, std::string>::iterator it = handler.getDocumentsMap().find("first document title");
    EXPECT_NE(it, handler.getDocumentsMap().end());
    EXPECT_STREQ(handler.getDocumentsMap().at("first document title").c_str(), "first document text");
    it = handler.getDocumentsMap().find("second document title");
    EXPECT_NE(it, handler.getDocumentsMap().end());
    EXPECT_STREQ(handler.getDocumentsMap().at("second document title").c_str(), "second document text");
}

TEST_F(RerankHandlerTest, ValidRequestDocumentsList) {
    json = R"({
    "model": "model",
    "query": "query",
    "documents": [
        "first document",
        "second document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    EXPECT_STREQ(handler.getModel().c_str(), "model");
    EXPECT_STREQ(handler.getQuery().c_str(), "query");
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 1);
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    ASSERT_EQ(handler.getDocumentsList().size(), 2);
    EXPECT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    EXPECT_STREQ(handler.getDocumentsList()[1].c_str(), "second document");
}

TEST_F(RerankHandlerTest, DocumentsArrayMixedElementTypes) {
    json = R"({
    "model": "model",
    "query": "query",
    "documents": [
        "first document",
        {
            "title": "second document title",
            "text": "second document text"
        }
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("all documents have to be the same type (string or objects)"));
}

TEST_F(RerankHandlerTest, InvalidDocuments) {
    json = R"({
    "model": "model",
    "query": "query",
    "documents": "INVALID"
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("documents is not an array"));
}

TEST_F(RerankHandlerTest, InvalidDocumentsElement) {
    json = R"({
    "model": "model",
    "query": "query",
    "documents": [1,2,3]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("documents array element is neither string nor object"));
}

TEST_F(RerankHandlerTest, ValidTopN) {
    json = R"({
    "model": "model",
    "query": "query",
    "top_n": 1,
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    EXPECT_STREQ(handler.getModel().c_str(), "model");
    EXPECT_STREQ(handler.getQuery().c_str(), "query");
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getDocumentsList().size(), 1);
    ASSERT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 1);
}

TEST_F(RerankHandlerTest, InvalidTopN) {
    json = R"({
    "model": "model",
    "query": "query",
    "top_n": "INVALID",
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("top_n accepts integer values"));
}

TEST_F(RerankHandlerTest, ValidRankFields) {
    json = R"({
    "model": "model",
    "query": "query",
    "rank_fields": ["first", "second"],
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    EXPECT_STREQ(handler.getModel().c_str(), "model");
    EXPECT_STREQ(handler.getQuery().c_str(), "query");
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 1);
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getDocumentsList().size(), 1);
    ASSERT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    ASSERT_TRUE(handler.getRankFields().has_value());
    ASSERT_EQ(handler.getRankFields().value().size(), 2);
    EXPECT_STREQ(handler.getRankFields().value()[0].c_str(), "first");
    EXPECT_STREQ(handler.getRankFields().value()[1].c_str(), "second");
}

TEST_F(RerankHandlerTest, InvalidRankFields) {
    json = R"({
    "model": "model",
    "query": "query",
    "rank_fields": "INVALID",
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("rank_fields is not an array"));
}

TEST_F(RerankHandlerTest, InvalidRankFieldsElement) {
    json = R"({
    "model": "model",
    "query": "query",
    "rank_fields": [1],
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("rank_fields array element is not a string"));
}

TEST_F(RerankHandlerTest, ValidReturnDocuments) {
    json = R"({
    "model": "model",
    "query": "query",
    "return_documents": true,
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    EXPECT_STREQ(handler.getModel().c_str(), "model");
    EXPECT_STREQ(handler.getQuery().c_str(), "query");
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 1);
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getDocumentsList().size(), 1);
    ASSERT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    ASSERT_TRUE(handler.getReturnDocuments().has_value());
    EXPECT_TRUE(handler.getReturnDocuments().value());
}

TEST_F(RerankHandlerTest, InvalidReturnDocuments) {
    json = R"({
    "model": "model",
    "query": "query",
    "return_documents": "INVALID",
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("return_documents accepts boolean values"));
}

TEST_F(RerankHandlerTest, ValidMaxChunksPerDoc) {
    json = R"({
    "model": "model",
    "query": "query",
    "max_chunks_per_doc": 2,
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::OkStatus());
    EXPECT_STREQ(handler.getModel().c_str(), "model");
    EXPECT_STREQ(handler.getQuery().c_str(), "query");
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 1);
    EXPECT_EQ(handler.getDocumentsList().size(), 1);
    ASSERT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    ASSERT_TRUE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getMaxChunksPerDoc().value(), 2);
}

TEST_F(RerankHandlerTest, InvalidMaxChunksPerDoc) {
    json = R"({
    "model": "model",
    "query": "query",
    "max_chunks_per_doc": "INVALID",
    "documents": [
        "first document"
    ]
    })";

    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    auto status = handler.parseRequest();
    EXPECT_EQ(status, absl::InvalidArgumentError("max_chunks_per_doc accepts integer values"));
}
