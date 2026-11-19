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
#include <memory>
#include <string>
#include <unordered_set>

#include <gtest/gtest.h>

#include "src/graph_metadata.hpp"
#include "src/mediapipe_internal/mediapipegraphconfig.hpp"
#include "src/port/rapidjson_document.hpp"
#include "src/servable_definition.hpp"
#include "src/status.hpp"
#include "constructor_enabled_model_manager.hpp"
#include "test_utils.hpp"

using namespace ovms;

static std::string toJson(const GraphMetadata& metadata) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    metadata.writeObject(writer);
    return buffer.GetString();
}

TEST(GraphMetadataTest, EmptySerializesToEmptyObject) {
    GraphMetadata metadata;
    EXPECT_TRUE(metadata.empty());
    EXPECT_EQ(toJson(metadata), "{}");
}

TEST(GraphMetadataTest, ScalarsAreSerialized) {
    GraphMetadata metadata;
    EXPECT_TRUE(metadata.setInt("max_model_len", 4096));
    EXPECT_TRUE(metadata.setBool("flag", true));
    EXPECT_TRUE(metadata.setDouble("ratio", 0.5));
    EXPECT_TRUE(metadata.setString("name", "model"));
    EXPECT_FALSE(metadata.empty());

    rapidjson::Document doc;
    ASSERT_FALSE(doc.Parse(toJson(metadata).c_str()).HasParseError());
    ASSERT_TRUE(doc["max_model_len"].IsInt64());
    EXPECT_EQ(doc["max_model_len"].GetInt64(), 4096);
    ASSERT_TRUE(doc["flag"].IsBool());
    EXPECT_TRUE(doc["flag"].GetBool());
    ASSERT_TRUE(doc["ratio"].IsDouble());
    EXPECT_DOUBLE_EQ(doc["ratio"].GetDouble(), 0.5);
    ASSERT_TRUE(doc["name"].IsString());
    EXPECT_STREQ(doc["name"].GetString(), "model");
}

TEST(GraphMetadataTest, NestedObjectIsSerialized) {
    auto inner = std::make_shared<GraphMetadata>();
    inner->setInt("depth", 2);
    auto middle = std::make_shared<GraphMetadata>();
    middle->setObject("inner", inner);
    GraphMetadata metadata;
    metadata.setObject("middle", middle);
    metadata.setObject("missing", nullptr);

    rapidjson::Document doc;
    ASSERT_FALSE(doc.Parse(toJson(metadata).c_str()).HasParseError());
    ASSERT_TRUE(doc["middle"]["inner"].IsObject());
    EXPECT_EQ(doc["middle"]["inner"]["depth"].GetInt64(), 2);
    EXPECT_TRUE(doc["missing"].IsNull());
}

TEST(GraphMetadataTest, NestedObjectIsCopiedOnInsert) {
    auto inner = std::make_shared<GraphMetadata>();
    inner->setInt("depth", 2);
    auto middle = std::make_shared<GraphMetadata>();
    middle->setObject("inner", inner);
    GraphMetadata metadata;
    metadata.setObject("middle", middle);
    const std::string before = toJson(metadata);

    middle->setInt("added_to_child", 1);
    inner->setInt("added_to_grandchild", 1);

    EXPECT_EQ(toJson(metadata), before);
}

TEST(GraphMetadataTest, KeysWithEmbeddedNulAreNotTruncated) {
    GraphMetadata metadata;
    const std::string keyWithNul("a\0b", 3);
    EXPECT_TRUE(metadata.setInt("a", 1));
    EXPECT_TRUE(metadata.setInt(keyWithNul, 2));
    const std::string json = toJson(metadata);
    EXPECT_EQ(json, std::string("{\"a\":1,\"a\\u0000b\":2}"));
}

TEST(GraphMetadataTest, FirstValueOfDuplicateKeyWins) {
    GraphMetadata metadata;
    EXPECT_TRUE(metadata.setInt("key", 1));
    EXPECT_FALSE(metadata.setInt("key", 2));
    EXPECT_FALSE(metadata.setString("key", "other"));
    const auto* value = metadata.find("key");
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(std::get<int64_t>(*value), 1);
    EXPECT_EQ(metadata.find("absent"), nullptr);
}

TEST(GraphMetadataTest, WriteMembersSkipsRequestedKeys) {
    GraphMetadata metadata;
    metadata.setString("id", "overridden");
    metadata.setInt("max_model_len", 8);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.String("id");
    writer.String("original");
    metadata.writeMembers(writer, std::unordered_set<std::string>{"id"});
    writer.EndObject();

    rapidjson::Document doc;
    ASSERT_FALSE(doc.Parse(buffer.GetString()).HasParseError());
    EXPECT_STREQ(doc["id"].GetString(), "original");
    EXPECT_EQ(doc["max_model_len"].GetInt64(), 8);
}

TEST(GraphMetadataTest, ServableDefinitionHasNoMetadataByDefault) {
    struct MinimalDefinition : ServableDefinition {
        std::string name = "minimal";
        const std::string& getName() const override { return name; }
    };
    MinimalDefinition definition;
    EXPECT_EQ(definition.getMetadata(), nullptr);
}

TEST(GraphMetadataTest, GraphWithoutMetadataProvidingNodesHasEmptyMetadata) {
    ConstructorEnabledModelManager manager;
    const std::string testPbtxt = R"(
        input_stream: "in"
        output_stream: "out"
    )";
    MediapipeGraphConfig mgc{"metadataDummy", "", ""};
    DummyMediapipeGraphDefinition definition("metadataDummy", mgc, testPbtxt, nullptr);
    ASSERT_EQ(definition.getMetadata(), nullptr);
    ASSERT_EQ(definition.validateForConfigFileExistence(), StatusCode::OK);
    ASSERT_EQ(definition.validateForConfigLoadablenessPublic(), StatusCode::OK);
    ASSERT_EQ(definition.initializeNodes(), StatusCode::OK);
    auto metadata = definition.getMetadata();
    ASSERT_NE(metadata, nullptr);
    EXPECT_TRUE(metadata->empty());
}
