//*****************************************************************************
// Copyright 2021 Intel Corporation
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

#include "../logging.hpp"
#include "../nodesessionmetadata.hpp"

using namespace ovms;

using testing::_;
using testing::HasSubstr;
using testing::Not;
using testing::Return;

class NodeSessionMetadataTest : public ::testing::Test {
    NodeSessionMetadata meta1;
};

TEST_F(NodeSessionMetadataTest, GenerateSessionKeyWhenNoSubsessions) {
    NodeSessionMetadata meta;
    EXPECT_EQ(meta.getSessionKey(), "");
}

TEST_F(NodeSessionMetadataTest, GenerateSubsession) {
    NodeSessionMetadata meta;
    auto demultiplexedMetas = meta.generateSubsessions("request", 2);
    ASSERT_EQ(demultiplexedMetas.size(), 2);
    EXPECT_EQ(demultiplexedMetas[0].getSessionKey(), "request_0");
    EXPECT_EQ(demultiplexedMetas[1].getSessionKey(), "request_1");
}

TEST_F(NodeSessionMetadataTest, GenerateTwoLevelsOfSubsession) {
    const uint firstLevelDemultiplexSize = 3;
    const uint secondLevelDemultiplexSize = 2;
    NodeSessionMetadata meta;
    auto demultiplexedMetas = meta.generateSubsessions("request", firstLevelDemultiplexSize);
    ASSERT_EQ(demultiplexedMetas.size(), firstLevelDemultiplexSize);
    std::vector<NodeSessionMetadata> secondLevelMetas(firstLevelDemultiplexSize * secondLevelDemultiplexSize);
    for (size_t demMetaId = 0; demMetaId != demultiplexedMetas.size(); ++demMetaId) {
        auto newLevelMetas = demultiplexedMetas[demMetaId].generateSubsessions("2ndDemultiplexer", secondLevelDemultiplexSize);
        std::move(newLevelMetas.begin(), newLevelMetas.end(), secondLevelMetas.begin() + demMetaId * secondLevelDemultiplexSize);
    }
    for (size_t demMetaId = 0; demMetaId != demultiplexedMetas.size(); ++demMetaId) {
        EXPECT_EQ(demultiplexedMetas[demMetaId].getSessionKey(), std::string("request_") + std::to_string(demMetaId));
    }
    for (size_t demMetaId = 0; demMetaId != firstLevelDemultiplexSize; ++demMetaId) {
        for (size_t demMetaLev2Id = 0; demMetaLev2Id != secondLevelDemultiplexSize; ++demMetaLev2Id) {
            auto hash = secondLevelMetas[demMetaLev2Id + demMetaId * secondLevelDemultiplexSize].getSessionKey();
            EXPECT_THAT(hash, HasSubstr(std::string("request_") + std::to_string(demMetaId)));
            EXPECT_THAT(hash, HasSubstr(std::string("2ndDemultiplexer_") + std::to_string(demMetaLev2Id)));
        }
    }
}

TEST_F(NodeSessionMetadataTest, GenerateThreeLevelsOfSubsession) {
    const uint firstLevelDemultiplexSize = 3;
    const uint secondLevelDemultiplexSize = 2;
    const uint thirdLevelDemultiplexSize = 4;
    NodeSessionMetadata meta;
    auto demultiplexedMetaLev3 = meta
                                     .generateSubsessions("request", firstLevelDemultiplexSize)[2]
                                     .generateSubsessions("extract1st", secondLevelDemultiplexSize)[0]
                                     .generateSubsessions("extract2nd", thirdLevelDemultiplexSize)[2];
    auto hash = demultiplexedMetaLev3.getSessionKey();
    EXPECT_THAT(hash, HasSubstr("request_2"));
    EXPECT_THAT(hash, HasSubstr("extract1st_0"));
    EXPECT_THAT(hash, HasSubstr("extract2nd_2"));
}

TEST_F(NodeSessionMetadataTest, GenerateSubsessionWithEmptyNameShouldThrow) {
    NodeSessionMetadata meta;
    EXPECT_THROW(meta.generateSubsessions("", 3), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CanGenerateEmptySubsession) {
    NodeSessionMetadata startMeta;
    auto meta = startMeta.generateSubsessions("someName", 0);
    EXPECT_EQ(meta.size(), 0) << meta[0].getSessionKey();
}

TEST_F(NodeSessionMetadataTest, GenerateTwoSubsessionsWithTheSameNameShouldThrow) {
    NodeSessionMetadata meta;
    auto newMetas = meta.generateSubsessions("request", 1);
    ASSERT_EQ(newMetas.size(), 1);
    EXPECT_THROW(newMetas[0].generateSubsessions("request", 12), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CollapseSubsession1Level) {
    const uint firstLevelDemultiplexSize = 3;
    const uint secondLevelDemultiplexSize = 2;
    const uint thirdLevelDemultiplexSize = 4;
    NodeSessionMetadata meta;
    auto demultiplexedMetaLev3 = meta
                                     .generateSubsessions("request", firstLevelDemultiplexSize)[2]
                                     .generateSubsessions("extract1st", secondLevelDemultiplexSize)[0]
                                     .generateSubsessions("extract2nd", thirdLevelDemultiplexSize)[2];
    auto hash = demultiplexedMetaLev3.getSessionKey();
    ASSERT_THAT(hash, HasSubstr("request_2"));
    ASSERT_THAT(hash, HasSubstr("extract1st_0"));
    ASSERT_THAT(hash, HasSubstr("extract2nd_2"));

    auto metaCollapsedOnExtract1st = demultiplexedMetaLev3.getCollapsedSessionMetadata({"extract1st"});
    auto hashCollapsed = metaCollapsedOnExtract1st.getSessionKey();
    ASSERT_THAT(hashCollapsed, HasSubstr("request_2"));
    ASSERT_THAT(hashCollapsed, Not(HasSubstr("extract1st_0")));
    ASSERT_THAT(hashCollapsed, HasSubstr("extract2nd_2"));
}

TEST_F(NodeSessionMetadataTest, CollapseSubsessions2LevelsAtOnce) {
    const uint firstLevelDemultiplexSize = 13;
    const uint secondLevelDemultiplexSize = 42;
    const uint thirdLevelDemultiplexSize = 666;
    NodeSessionMetadata meta;
    auto demultiplexedMetaLev3 = meta
                                     .generateSubsessions("request", firstLevelDemultiplexSize)[12]
                                     .generateSubsessions("extract1st", secondLevelDemultiplexSize)[32]
                                     .generateSubsessions("extract2nd", thirdLevelDemultiplexSize)[512];
    auto hash = demultiplexedMetaLev3.getSessionKey();
    ASSERT_THAT(hash, HasSubstr("request_12"));
    ASSERT_THAT(hash, HasSubstr("extract1st_32"));
    ASSERT_THAT(hash, HasSubstr("extract2nd_512"));

    auto metaCollapsed = demultiplexedMetaLev3.getCollapsedSessionMetadata({"extract1st", "extract2nd"});
    auto hashCollapsed = metaCollapsed.getSessionKey();
    ASSERT_THAT(hashCollapsed, HasSubstr("request_12"));
    ASSERT_THAT(hashCollapsed, Not(HasSubstr("extract1st")));
    ASSERT_THAT(hashCollapsed, Not(HasSubstr("extract2nd")));
}

TEST_F(NodeSessionMetadataTest, CollapsingNonExistingSubsessionShouldThrow) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0];
    EXPECT_THROW(subsessionMeta.getCollapsedSessionMetadata({"NonExistingSubsessionName"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CollapsingManySubsessionsButOneNonExistingShouldThrow) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0]
                              .generateSubsessions("anotherSession", 5)[1];
    EXPECT_THROW(subsessionMeta.getCollapsedSessionMetadata({"anotherSession", "NonExistingSubsessionName"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSubsessionKey) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0]
                              .generateSubsessions("anotherSession", 5)[1];
    auto hash = subsessionMeta.getSessionKey({"anotherSession"});
    ASSERT_THAT(hash, HasSubstr("request_0"));
    ASSERT_THAT(hash, Not(HasSubstr("anotherSession")));
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSeveralSubsessionsAtOnceKey) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0]
                              .generateSubsessions("anotherSession", 5)[1]
                              .generateSubsessions("yetAnotherSession", 3)[2];
    auto hash = subsessionMeta.getSessionKey({"anotherSession", "yetAnotherSession"});
    ASSERT_THAT(hash, HasSubstr("request"));
    ASSERT_THAT(hash, Not(HasSubstr("anotherSession")));
    ASSERT_THAT(hash, Not(HasSubstr("yetAnotherSession")));
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSubsessionKeyShouldThrowWhenNonExistingSubsession) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 2)[1];
    EXPECT_THROW(subsessionMeta.getSessionKey({"NonExistingSubsession"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSeveralSubsessionKeyShouldThrowWhenJustOneNonExisting) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 2)[1]
                              .generateSubsessions("anotherSession", 5)[1];
    EXPECT_THROW(subsessionMeta.getSessionKey({"anotherSession", "NonExistingSubsession"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, ReturnSubsessionSize) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 5)[0];
    EXPECT_EQ(subsessionMeta.getSubsessionSize("request"), 5);
}

TEST_F(NodeSessionMetadataTest, ReturnSubsessionsSizeForAllLevels) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 5)[0]
                              .generateSubsessions("extract1", 4)[0]
                              .generateSubsessions("extract2", 3)[0]
                              .generateSubsessions("extract3", 2)[0];
    EXPECT_EQ(subsessionMeta.getSubsessionSize("request"), 5);
    EXPECT_EQ(subsessionMeta.getSubsessionSize("extract1"), 4);
    EXPECT_EQ(subsessionMeta.getSubsessionSize("extract2"), 3);
    EXPECT_EQ(subsessionMeta.getSubsessionSize("extract3"), 2);
}

TEST_F(NodeSessionMetadataTest, GetSubsessionSizeShouldThrowWhenNonExistingSubsession) {
    NodeSessionMetadata meta;
    auto subsessionMeta = meta.generateSubsessions("request", 5)[0];
    EXPECT_THROW(subsessionMeta.getSubsessionSize("nonExisting"), std::logic_error);
}
