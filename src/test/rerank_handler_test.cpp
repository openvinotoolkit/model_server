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

class RerankHandlerDeserializationTest : public ::testing::Test {
protected:
    Document doc;
    std::string json;
    void SetUp() override {
    }
};

TEST_F(RerankHandlerDeserializationTest, ValidRequestDocumentsMap) {
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
    auto it = handler.getDocumentsMap().find("first document title");
    EXPECT_NE(it, handler.getDocumentsMap().end());
    EXPECT_STREQ(handler.getDocumentsMap().at("first document title").c_str(), "first document text");
    it = handler.getDocumentsMap().find("second document title");
    EXPECT_NE(it, handler.getDocumentsMap().end());
    EXPECT_STREQ(handler.getDocumentsMap().at("second document title").c_str(), "second document text");
}

TEST_F(RerankHandlerDeserializationTest, ValidRequestDocumentsList) {
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
    EXPECT_EQ(handler.getTopN().value(), 2);
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    ASSERT_EQ(handler.getDocumentsList().size(), 2);
    EXPECT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    EXPECT_STREQ(handler.getDocumentsList()[1].c_str(), "second document");
}

TEST_F(RerankHandlerDeserializationTest, DocumentsArrayMixedElementTypes) {
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

TEST_F(RerankHandlerDeserializationTest, InvalidJson) {
    json = R"({
    INVALID JSON
    })";

    ASSERT_TRUE(doc.Parse(json.c_str()).HasParseError());
    RerankHandler handler(doc);
    ASSERT_EQ(handler.parseRequest(), absl::InvalidArgumentError("Non-json request received in rerank calculator"));
}

TEST_F(RerankHandlerDeserializationTest, InvalidDocuments) {
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

TEST_F(RerankHandlerDeserializationTest, InvalidDocumentsElement) {
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

TEST_F(RerankHandlerDeserializationTest, ValidTopN) {
    json = R"({
    "model": "model",
    "query": "query",
    "top_n": 1,
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
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getDocumentsList().size(), 2);
    ASSERT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 1);
}

TEST_F(RerankHandlerDeserializationTest, TopNNull) {
    json = R"({
    "model": "model",
    "query": "query",
    "top_n": null,
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
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
    ASSERT_FALSE(handler.getRankFields().has_value());
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
    EXPECT_EQ(handler.getDocumentsList().size(), 2);
    ASSERT_EQ(handler.getDocumentsMap().size(), 0);
    EXPECT_STREQ(handler.getDocumentsList()[0].c_str(), "first document");
    ASSERT_TRUE(handler.getTopN().has_value());
    EXPECT_EQ(handler.getTopN().value(), 2);
}

TEST_F(RerankHandlerDeserializationTest, InvalidTopN) {
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

TEST_F(RerankHandlerDeserializationTest, ValidRankFields) {
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

TEST_F(RerankHandlerDeserializationTest, RankFieldsNull) {
    json = R"({
    "model": "model",
    "query": "query",
    "rank_fields": null,
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
    ASSERT_FALSE(handler.getRankFields().has_value());
}

TEST_F(RerankHandlerDeserializationTest, InvalidRankFields) {
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

TEST_F(RerankHandlerDeserializationTest, InvalidRankFieldsElement) {
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

TEST_F(RerankHandlerDeserializationTest, ValidReturnDocuments) {
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

TEST_F(RerankHandlerDeserializationTest, ReturnDocumentsNull) {
    json = R"({
    "model": "model",
    "query": "query",
    "return_documents": null,
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
    ASSERT_FALSE(handler.getReturnDocuments().has_value());
}

TEST_F(RerankHandlerDeserializationTest, InvalidReturnDocuments) {
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

TEST_F(RerankHandlerDeserializationTest, ValidMaxChunksPerDoc) {
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

TEST_F(RerankHandlerDeserializationTest, MaxChunksPerDocNull) {
    json = R"({
    "model": "model",
    "query": "query",
    "max_chunks_per_doc": null,
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
    ASSERT_FALSE(handler.getMaxChunksPerDoc().has_value());
}

TEST_F(RerankHandlerDeserializationTest, InvalidMaxChunksPerDoc) {
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

TEST(RerankHandlerSerializationTest, simplePostivie) {
    std::vector<float> scores = {5.36, 17.21, 3.01, 22.33, 9.4, 22.33};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "documents": ["unused", "unused", "unused", "unused", "unused", "unused"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    ASSERT_TRUE(handler.parseRequest().ok());
    StringBuffer buffer;
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_TRUE(status.ok());
    std::string expectedResponse = R"({"results":[{"index":3,"relevance_score":22.329999923706055},{"index":5,"relevance_score":22.329999923706055},{"index":1,"relevance_score":17.209999084472656},{"index":4,"relevance_score":9.399999618530273},{"index":0,"relevance_score":5.360000133514404},{"index":2,"relevance_score":3.009999990463257}]})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(RerankHandlerSerializationTest, positiveSmallNumbers) {
    std::vector<float> scores = {0.000000001, 0.000000002, 0.000000003};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "documents": ["unused", "unused", "unused"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    ASSERT_TRUE(handler.parseRequest().ok());
    StringBuffer buffer;
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_TRUE(status.ok());
    std::string expectedResponse = R"({"results":[{"index":2,"relevance_score":3.000000026176508e-9},{"index":1,"relevance_score":1.999999943436137e-9},{"index":0,"relevance_score":9.999999717180685e-10}]})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(RerankHandlerSerializationTest, positiveReturnDocumentsWithDocumentsList) {
    std::vector<float> scores = {5.36, 17.21, 3.01};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "return_documents": true,
    "documents": [ "first", "second", "third"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    StringBuffer buffer;
    ASSERT_TRUE(handler.parseRequest().ok());
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_TRUE(status.ok());
    std::string expectedResponse = R"({"results":[{"index":1,"relevance_score":17.209999084472656,"document":{"text":"second"}},{"index":0,"relevance_score":5.360000133514404,"document":{"text":"first"}},{"index":2,"relevance_score":3.009999990463257,"document":{"text":"third"}}]})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(RerankHandlerSerializationTest, positiveReturnDocumentsFalseWithDocumentsList) {
    std::vector<float> scores = {5.36, 17.21, 3.01};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "return_documents": false,
    "documents": [ "first", "second", "third"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    StringBuffer buffer;
    ASSERT_TRUE(handler.parseRequest().ok());
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_TRUE(status.ok());
    std::string expectedResponse = R"({"results":[{"index":1,"relevance_score":17.209999084472656},{"index":0,"relevance_score":5.360000133514404},{"index":2,"relevance_score":3.009999990463257}]})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(RerankHandlerSerializationTest, negativeReturnDocumentsWithDocumentsMap) {  // TODO add support for return documents for documents map
    std::vector<float> scores = {5.36, 17.21, 3.01, 22.33, 9.4, 22.33};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "return_documents": true,
    "documents": [
        {
        "title": "first document title",
        "text": "first document text"
        }
    ]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    ASSERT_TRUE(handler.parseRequest().ok());
    StringBuffer buffer;
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_FALSE(status.ok());
}

TEST(RerankHandlerSerializationTest, negativeReturnDocumentsWithDocumentsListWithLessDocumentsThanScores) {
    std::vector<float> scores = {5.36, 17.21, 3.01, 4};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "return_documents": true,
    "documents": [ "first", "second", "third"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    ASSERT_TRUE(handler.parseRequest().ok());
    StringBuffer buffer;
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_FALSE(status.ok());
}

TEST(RerankHandlerSerializationTest, positiveTopN) {
    std::vector<float> scores = {5.36, 17.21, 3.01, 22.33, 9.4, 22.33};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "top_n": 3,
    "documents": ["unused", "unused", "unused", "unused", "unused", "unused"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    ASSERT_TRUE(handler.parseRequest().ok());
    StringBuffer buffer;
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_TRUE(status.ok());
    std::string expectedResponse = R"({"results":[{"index":3,"relevance_score":22.329999923706055},{"index":5,"relevance_score":22.329999923706055},{"index":1,"relevance_score":17.209999084472656}]})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}

TEST(RerankHandlerSerializationTest, positiveTopNHigherThanNumberOfDocuments) {
    std::vector<float> scores = {5.36, 17.21, 3.01, 22.33, 9.4, 22.33};
    std::string json = R"({
    "model": "model",
    "query": "query",
    "top_n": 10,
    "documents": ["unused", "unused", "unused", "unused", "unused", "unused"]
    })";

    Document d;
    ASSERT_FALSE(d.Parse(json.c_str()).HasParseError());
    RerankHandler handler(d);
    ASSERT_TRUE(handler.parseRequest().ok());
    StringBuffer buffer;
    auto status = handler.parseResponse(buffer, scores);
    EXPECT_TRUE(status.ok());
    std::string expectedResponse = R"({"results":[{"index":3,"relevance_score":22.329999923706055},{"index":5,"relevance_score":22.329999923706055},{"index":1,"relevance_score":17.209999084472656},{"index":4,"relevance_score":9.399999618530273},{"index":0,"relevance_score":5.360000133514404},{"index":2,"relevance_score":3.009999990463257}]})";
    EXPECT_STREQ(buffer.GetString(), expectedResponse.c_str());
}
