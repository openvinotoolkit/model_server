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

#include "../dags/nodesessionmetadata.hpp"
#include "../logging.hpp"
#include "test_utils.hpp"

using namespace ovms;

using testing::_;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::Not;
using testing::Return;

class NodeSessionMetadataTest : public ::testing::Test {};

TEST_F(NodeSessionMetadataTest, GenerateSessionKeyWhenNoSubsessions) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    EXPECT_EQ(meta.getSessionKey(), "");
}

TEST_F(NodeSessionMetadataTest, GenerateSubsession) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto demultiplexedMetas = meta.generateSubsessions("request", 2);
    ASSERT_EQ(demultiplexedMetas.size(), 2);
    EXPECT_EQ(demultiplexedMetas[0].getSessionKey(), "request_0");
    EXPECT_EQ(demultiplexedMetas[1].getSessionKey(), "request_1");
}

TEST_F(NodeSessionMetadataTest, GenerateTwoLevelsOfSubsession) {
    const uint32_t firstLevelDemultiplexSize = 3;
    const uint32_t secondLevelDemultiplexSize = 2;
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto demultiplexedMetas = meta.generateSubsessions("request", firstLevelDemultiplexSize);
    ASSERT_EQ(demultiplexedMetas.size(), firstLevelDemultiplexSize);
    std::vector<NodeSessionMetadata> secondLevelMetas(firstLevelDemultiplexSize * secondLevelDemultiplexSize, NodeSessionMetadata{DEFAULT_TEST_CONTEXT});
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
    const uint32_t firstLevelDemultiplexSize = 3;
    const uint32_t secondLevelDemultiplexSize = 2;
    const uint32_t thirdLevelDemultiplexSize = 4;
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
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
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    EXPECT_THROW(meta.generateSubsessions("", 3), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CanGenerateEmptySubsession) {
    NodeSessionMetadata startMeta{DEFAULT_TEST_CONTEXT};
    auto meta = startMeta.generateSubsessions("someName", 0);
    EXPECT_EQ(meta.size(), 0) << meta[0].getSessionKey();
}

TEST_F(NodeSessionMetadataTest, GenerateTwoSubsessionsWithTheSameNameShouldThrow) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto newMetas = meta.generateSubsessions("request", 1);
    ASSERT_EQ(newMetas.size(), 1);
    EXPECT_THROW(newMetas[0].generateSubsessions("request", 12), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CollapseSubsession1Level) {
    const uint32_t firstLevelDemultiplexSize = 3;
    const uint32_t secondLevelDemultiplexSize = 2;
    const uint32_t thirdLevelDemultiplexSize = 4;
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto demultiplexedMetaLev3 = meta
                                     .generateSubsessions("request", firstLevelDemultiplexSize)[2]
                                     .generateSubsessions("extract1st", secondLevelDemultiplexSize)[0]
                                     .generateSubsessions("extract2nd", thirdLevelDemultiplexSize)[2];
    auto hash = demultiplexedMetaLev3.getSessionKey();
    ASSERT_THAT(hash, HasSubstr("request_2"));
    ASSERT_THAT(hash, HasSubstr("extract1st_0"));
    ASSERT_THAT(hash, HasSubstr("extract2nd_2"));
    NodeSessionMetadata metaCollapsedOnExtract1st{DEFAULT_TEST_CONTEXT};
    CollapseDetails collapsingDetails;
    std::tie(metaCollapsedOnExtract1st, collapsingDetails) = demultiplexedMetaLev3.getCollapsedSessionMetadata({"extract2nd"});
    auto hashCollapsed = metaCollapsedOnExtract1st.getSessionKey();
    // need to ensure that generated collapsed session key before collapsing and after are the same
    EXPECT_EQ(hashCollapsed, demultiplexedMetaLev3.getSessionKey({std::string("extract2nd")}));

    ASSERT_THAT(hashCollapsed, HasSubstr("request_2"));
    ASSERT_THAT(hashCollapsed, HasSubstr("extract1st_0"));
    ASSERT_THAT(hashCollapsed, Not(HasSubstr("extract2nd_2")));
    ASSERT_EQ(collapsingDetails.collapsedSessionNames.size(), 1);
    ASSERT_EQ(collapsingDetails.collapsedSessionSizes.size(), 1);
    ASSERT_EQ(collapsingDetails.collapsedSessionNames[0], "extract2nd");
    ASSERT_EQ(collapsingDetails.collapsedSessionSizes[0], thirdLevelDemultiplexSize);
}

TEST_F(NodeSessionMetadataTest, CollapseSubsession1LevelNotInLIFOOrderShouldThrow) {
    const uint32_t firstLevelDemultiplexSize = 3;
    const uint32_t secondLevelDemultiplexSize = 2;
    const uint32_t thirdLevelDemultiplexSize = 4;
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto demultiplexedMetaLev3 = meta
                                     .generateSubsessions("request", firstLevelDemultiplexSize)[2]
                                     .generateSubsessions("extract1st", secondLevelDemultiplexSize)[0]
                                     .generateSubsessions("extract2nd", thirdLevelDemultiplexSize)[2];
    auto hash = demultiplexedMetaLev3.getSessionKey();
    ASSERT_THAT(hash, HasSubstr("request_2"));
    ASSERT_THAT(hash, HasSubstr("extract1st_0"));
    ASSERT_THAT(hash, HasSubstr("extract2nd_2"));
    NodeSessionMetadata metaCollapsedOnExtract1st{DEFAULT_TEST_CONTEXT};
    CollapseDetails collapsingDetails;
    EXPECT_THROW(demultiplexedMetaLev3.getCollapsedSessionMetadata({"extract1st"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CollapseSubsessions2LevelsAtOnce) {
    const uint32_t firstLevelDemultiplexSize = 13;
    const uint32_t secondLevelDemultiplexSize = 42;
    const uint32_t thirdLevelDemultiplexSize = 666;
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto demultiplexedMetaLev3 = meta
                                     .generateSubsessions("request", firstLevelDemultiplexSize)[12]
                                     .generateSubsessions("extract1st", secondLevelDemultiplexSize)[32]
                                     .generateSubsessions("extract2nd", thirdLevelDemultiplexSize)[512];
    auto hash = demultiplexedMetaLev3.getSessionKey();
    ASSERT_THAT(hash, HasSubstr("request_12"));
    ASSERT_THAT(hash, HasSubstr("extract1st_32"));
    ASSERT_THAT(hash, HasSubstr("extract2nd_512"));

    NodeSessionMetadata metaCollapsed{DEFAULT_TEST_CONTEXT};
    CollapseDetails collapsingDetails;
    std::tie(metaCollapsed, collapsingDetails) = demultiplexedMetaLev3.getCollapsedSessionMetadata({"extract1st", "extract2nd"});
    auto hashCollapsed = metaCollapsed.getSessionKey();
    ASSERT_THAT(hashCollapsed, HasSubstr("request_12"));
    ASSERT_THAT(hashCollapsed, Not(HasSubstr("extract1st")));
    ASSERT_THAT(hashCollapsed, Not(HasSubstr("extract2nd")));
    ASSERT_EQ(collapsingDetails.collapsedSessionNames.size(), 2);
    ASSERT_EQ(collapsingDetails.collapsedSessionSizes.size(), 2);
    EXPECT_THAT(collapsingDetails.collapsedSessionNames,
        ElementsAre("extract1st", "extract2nd"));
    EXPECT_THAT(collapsingDetails.collapsedSessionSizes,
        ElementsAre(secondLevelDemultiplexSize, thirdLevelDemultiplexSize));
}

TEST_F(NodeSessionMetadataTest, CollapsingNonExistingSubsessionShouldThrow) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0];
    EXPECT_THROW(subsessionMeta.getCollapsedSessionMetadata({"NonExistingSubsessionName"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, CollapsingManySubsessionsButOneNonExistingShouldThrow) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0]
                              .generateSubsessions("anotherSession", 5)[1];
    EXPECT_THROW(subsessionMeta.getCollapsedSessionMetadata({"anotherSession", "NonExistingSubsessionName"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSubsessionKey) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0]
                              .generateSubsessions("anotherSession", 5)[1];
    auto hash = subsessionMeta.getSessionKey({"anotherSession"});
    ASSERT_THAT(hash, HasSubstr("request_0"));
    ASSERT_THAT(hash, Not(HasSubstr("anotherSession")));
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSeveralSubsessionsAtOnceKey) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 2)[0]
                              .generateSubsessions("anotherSession", 5)[1]
                              .generateSubsessions("yetAnotherSession", 3)[2];
    auto hash = subsessionMeta.getSessionKey({"anotherSession", "yetAnotherSession"});
    ASSERT_THAT(hash, HasSubstr("request"));
    ASSERT_THAT(hash, Not(HasSubstr("anotherSession")));
    ASSERT_THAT(hash, Not(HasSubstr("yetAnotherSession")));
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSubsessionKeyShouldThrowWhenNonExistingSubsession) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 2)[1];
    EXPECT_THROW(subsessionMeta.getSessionKey({"NonExistingSubsession"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, GenerateCollapsedSeveralSubsessionKeyShouldThrowWhenJustOneNonExisting) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 2)[1]
                              .generateSubsessions("anotherSession", 5)[1];
    EXPECT_THROW(subsessionMeta.getSessionKey({"anotherSession", "NonExistingSubsession"}), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, ReturnSubsessionSize) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 5)[0];
    EXPECT_EQ(subsessionMeta.getSubsessionSize("request"), 5);
}

TEST_F(NodeSessionMetadataTest, ReturnSubsessionsSizeForAllLevels) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
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
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    auto subsessionMeta = meta.generateSubsessions("request", 5)[0];
    EXPECT_THROW(subsessionMeta.getSubsessionSize("nonExisting"), std::logic_error);
}

TEST_F(NodeSessionMetadataTest, GetShardIdNoSubsession) {
    NodeSessionMetadata meta{DEFAULT_TEST_CONTEXT};
    EXPECT_EQ(meta.getShardId(), 0);
}
TEST_F(NodeSessionMetadataTest, GetShardId1SubsessionLevel) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize = 13;
    const std::string subsessionName = "subsession";
    auto subsessions = metaStart.generateSubsessions(subsessionName, subsessionSize);
    ASSERT_EQ(subsessions.size(), subsessionSize);
    for (size_t i = 0; i < subsessions.size(); ++i) {
        EXPECT_EQ(subsessions[i].getShardId(), 0);
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId1SubsessionLevelCollapsing) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize = 13;
    const std::string subsessionName = "subsession";
    auto subsessions = metaStart.generateSubsessions(subsessionName, subsessionSize);
    ASSERT_EQ(subsessions.size(), subsessionSize);
    for (size_t i = 0; i < subsessions.size(); ++i) {
        EXPECT_EQ(subsessions[i].getShardId({subsessionName}), i);
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId2SubsessionLevels) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1st = 13;
    uint32_t subsessionSize2nd = 9;
    const std::string subsessionName1st = "subsession";
    const std::string subsessionName2nd = "subsession2";
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1st, subsessionSize1st);
    auto subsessionsLevel2 = subsessionsLevel1[4].generateSubsessions(subsessionName2nd, subsessionSize2nd);
    for (size_t i = 0; i < subsessionsLevel2.size(); ++i) {
        EXPECT_EQ(subsessionsLevel2[i].getShardId(), 0);
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId2SubsessionLevelsCollapse1) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1st = 13;
    uint32_t subsessionSize2nd = 9;
    const std::string subsessionName1st = "subsession";
    const std::string subsessionName2nd = "subsession2";
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1st, subsessionSize1st);
    auto subsessionsLevel2 = subsessionsLevel1[4].generateSubsessions(subsessionName2nd, subsessionSize2nd);
    for (size_t i = 0; i < subsessionsLevel2.size(); ++i) {
        EXPECT_EQ(subsessionsLevel2[i].getShardId({subsessionName2nd}), i);
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId2SubsessionLevelsCollapse1NotInOrderShouldThrow) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1st = 13;
    uint32_t subsessionSize2nd = 9;
    const std::string subsessionName1st = "subsession";
    const std::string subsessionName2nd = "subsession2";
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1st, subsessionSize1st);
    auto subsessionsLevel2 = subsessionsLevel1[4].generateSubsessions(subsessionName2nd, subsessionSize2nd);
    for (size_t i = 0; i < subsessionsLevel2.size(); ++i) {
        EXPECT_THROW(subsessionsLevel2[i].getShardId({subsessionName1st}), std::logic_error);
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId2SubsessionLevelsCollapse2) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1st = 13;
    uint32_t subsessionSize2nd = 9;
    const std::string subsessionName1st = "subsession";
    const std::string subsessionName2nd = "subsession2";
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1st, subsessionSize1st);
    const int subsessionLev1Index = 4;
    auto subsessionsLevel2 = subsessionsLevel1[subsessionLev1Index].generateSubsessions(subsessionName2nd, subsessionSize2nd);
    for (size_t i = 0; i < subsessionsLevel2.size(); ++i) {
        EXPECT_EQ(subsessionsLevel2[i].getShardId({subsessionName2nd, subsessionName1st}), subsessionLev1Index * subsessionSize2nd + i);
    }
}
TEST_F(NodeSessionMetadataTest, GetShardId2SubsessionLevelsCollapse3ShouldThrow) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1st = 13;
    uint32_t subsessionSize2nd = 9;
    const std::string subsessionName1st = "subsession";
    const std::string subsessionName2nd = "subsession2";
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1st, subsessionSize1st);
    const int subsessionLev1Index = 4;
    auto subsessionsLevel2 = subsessionsLevel1[subsessionLev1Index].generateSubsessions(subsessionName2nd, subsessionSize2nd);
    for (size_t i = 0; i < subsessionsLevel2.size(); ++i) {
        EXPECT_THROW(subsessionsLevel2[i].getShardId({subsessionName2nd, subsessionName1st, std::string("NON_EXISTING_LEVEL")}), std::logic_error);
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId4SubsessionLevelsCollapse3) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1 = 13;
    uint32_t subsessionSize2 = 9;
    uint32_t subsessionSize3 = 7;
    uint32_t subsessionSize4 = 5;
    const std::string subsessionName1 = "subsession1";
    const std::string subsessionName2 = "subsession2";
    const std::string subsessionName3 = "subsession3";
    const std::string subsessionName4 = "subsession4";
    const int subsessionLev1Index = 4;
    const int subsessionLev2Index = 6;
    const int subsessionLev3Index = 3;
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1, subsessionSize1);
    auto subsessionsLevel2 = subsessionsLevel1[subsessionLev1Index].generateSubsessions(subsessionName2, subsessionSize2);
    auto subsessionsLevel3 = subsessionsLevel2[subsessionLev2Index].generateSubsessions(subsessionName3, subsessionSize3);
    auto subsessionsLevel4 = subsessionsLevel3[subsessionLev3Index].generateSubsessions(subsessionName4, subsessionSize4);
    for (size_t i = 0; i < subsessionsLevel4.size(); ++i) {
        EXPECT_EQ(subsessionsLevel4[i].getShardId({subsessionName2, subsessionName3, subsessionName4}),
            i + subsessionSize4 * (subsessionLev3Index + subsessionSize3 * (subsessionLev2Index)));
    }
}

TEST_F(NodeSessionMetadataTest, GetShardId4SubsessionLevelsCollapse1) {
    NodeSessionMetadata metaStart{DEFAULT_TEST_CONTEXT};
    uint32_t subsessionSize1 = 13;
    uint32_t subsessionSize2 = 9;
    uint32_t subsessionSize3 = 7;
    uint32_t subsessionSize4 = 5;
    const std::string subsessionName1 = "subsession1";
    const std::string subsessionName2 = "subsession2";
    const std::string subsessionName3 = "subsession3";
    const std::string subsessionName4 = "subsession4";
    const int subsessionLev1Index = 4;
    const int subsessionLev2Index = 6;
    const int subsessionLev3Index = 3;
    auto subsessionsLevel1 = metaStart.generateSubsessions(subsessionName1, subsessionSize1);
    auto subsessionsLevel2 = subsessionsLevel1[subsessionLev1Index].generateSubsessions(subsessionName2, subsessionSize2);
    auto subsessionsLevel3 = subsessionsLevel2[subsessionLev2Index].generateSubsessions(subsessionName3, subsessionSize3);
    auto subsessionsLevel4 = subsessionsLevel3[subsessionLev3Index].generateSubsessions(subsessionName4, subsessionSize4);
    for (size_t i = 0; i < subsessionsLevel4.size(); ++i) {
        EXPECT_EQ(subsessionsLevel4[i].getShardId({subsessionName4}),
            i);
    }
}
