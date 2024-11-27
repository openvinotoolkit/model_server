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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../rerank/rerank_utils.hpp"

using namespace ovms;

using ::testing::ElementsAre;

class RerankChunkingTest : public ::testing::Test {
protected:
    size_t max_allowed_chunks = 10;
};

TEST_F(RerankChunkingTest, ChunkingTest) {
    // w=4, h=6
    std::vector<int64_t> input_ids_data = {
        101, 102, 103, 104, 1, 1,      // 4 tokens
        105, 106, 107, 108, 109, 110,  // 6 tokens
        110, 1, 1, 1, 1, 1,            // 1 token
        1, 1, 1, 1, 1, 1,              // 0 tokens
    };
    std::vector<int64_t> attention_mask_data = {
        1, 1, 1, 1, 0, 0,  // 4 tokens
        1, 1, 1, 1, 1, 1,  // 6 tokens
        1, 0, 0, 0, 0, 0,  // 1 token
        0, 0, 0, 0, 0, 0,  // 0 tokens
    };
    size_t max_tokens_per_chunk = 3;
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i64, ov::Shape{4, 6}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i64, ov::Shape{4, 6}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::OkStatus());

    ASSERT_EQ(out_input_ids.get_shape().size(), 2);
    ASSERT_EQ(out_attention_mask.get_shape().size(), 2);

    ASSERT_EQ(out_input_ids.get_shape()[0], 6);
    ASSERT_EQ(out_input_ids.get_shape()[1], max_tokens_per_chunk);
    ASSERT_EQ(out_attention_mask.get_shape()[0], 6);
    ASSERT_EQ(out_attention_mask.get_shape()[1], max_tokens_per_chunk);

    std::vector<int64_t> expected_input_ids_data = {
        101, 102, 103,  // 3 tokens
        104, 1, 1,      // 1 token
        105, 106, 107,  // 3 tokens
        108, 109, 110,  // 3 tokens
        110, 1, 1,      // 1 token
        1, 1, 1,        // 0 tokens
    };
    std::vector<int64_t> expected_attention_mask_data = {
        1, 1, 1,  // 3 tokens
        1, 0, 0,  // 1 token
        1, 1, 1,  // 3 tokens
        1, 1, 1,  // 3 tokens
        1, 0, 0,  // 1 token
        0, 0, 0,  // 0 tokensgi
    };

    ASSERT_EQ(std::memcmp(
                  out_input_ids.data(),
                  expected_input_ids_data.data(),
                  out_input_ids.get_byte_size()),
        0);

    ASSERT_EQ(std::memcmp(
                  out_attention_mask.data(),
                  expected_attention_mask_data.data(),
                  out_attention_mask.get_byte_size()),
        0);

    ASSERT_THAT(chunk_mapping, ElementsAre(0, 0, 1, 1, 2, 3));
}

TEST_F(RerankChunkingTest, NoChunkingNeededTest) {
    // w=3, h=4
    std::vector<int64_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        105, 106, 107,  // 3 tokens
        110, 1, 1,      // 1 token
        1, 1, 1,        // 0 tokens
    };
    std::vector<int64_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
        1, 0, 0,  // 1 token
        0, 0, 0,  // 0 tokens
    };
    size_t max_tokens_per_chunk = 3;  // enough to fit all tokens
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i64, ov::Shape{4, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i64, ov::Shape{4, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::OkStatus());

    ASSERT_EQ(out_input_ids.get_shape().size(), 2);
    ASSERT_EQ(out_attention_mask.get_shape().size(), 2);

    ASSERT_EQ(out_input_ids.get_shape()[0], 4);
    ASSERT_EQ(out_input_ids.get_shape()[1], max_tokens_per_chunk);
    ASSERT_EQ(out_attention_mask.get_shape()[0], 4);
    ASSERT_EQ(out_attention_mask.get_shape()[1], max_tokens_per_chunk);

    ASSERT_EQ(std::memcmp(
                  out_input_ids.data(),
                  input_ids_data.data(),
                  out_input_ids.get_byte_size()),
        0);

    ASSERT_EQ(std::memcmp(
                  out_attention_mask.data(),
                  attention_mask_data.data(),
                  out_attention_mask.get_byte_size()),
        0);

    ASSERT_THAT(chunk_mapping, ElementsAre(0, 1, 2, 3));
}

TEST_F(RerankChunkingTest, InputIdsAndAttentionMaskShapesMismatchTest) {
    std::vector<int64_t> input_ids_data = {
        101, 102, 1,  // 2 tokens
    };
    std::vector<int64_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 3;  // enough to fit all tokens
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i64, ov::Shape{1, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i64, ov::Shape{2, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("input_ids and attention_mask shapes do not match"));
}

TEST_F(RerankChunkingTest, InputIdsAndAttentionMaskPrecisionMismatchTest) {
    std::vector<int64_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        101, 102, 103,  // 3 tokens
    };
    std::vector<int32_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 3;  // enough to fit all tokens
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i64, ov::Shape{2, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i32, ov::Shape{2, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("input_ids and attention_mask should have the same element type"));
}

TEST_F(RerankChunkingTest, InputIdsWrongPrecisionTest) {
    std::vector<int32_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        101, 102, 103,  // 3 tokens
    };
    std::vector<int32_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 3;  // enough to fit all tokens
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i32, ov::Shape{2, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i32, ov::Shape{2, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("input_ids and attention_mask should be int64 tensors"));
}

TEST_F(RerankChunkingTest, InputIdsWrongShapeTest) {
    std::vector<int32_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        101, 102, 103,  // 3 tokens
    };
    std::vector<int32_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 3;  // enough to fit all tokens
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i32, ov::Shape{1, 1, 6}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i32, ov::Shape{1, 1, 6}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("input_ids and attention_mask should be 2D tensors"));
}

TEST_F(RerankChunkingTest, NoSpaceLeftForChunkingTest) {
    std::vector<int32_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        101, 102, 103,  // 3 tokens
    };
    std::vector<int32_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 0;  // wrong
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i32, ov::Shape{2, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i32, ov::Shape{2, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        this->max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("no space left for chunks"));
}

TEST_F(RerankChunkingTest, MaxAllowedChunkExceededBeforeChunkingTest) {
    std::vector<int64_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        101, 102, 103,  // 3 tokens
    };
    std::vector<int64_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 3;  //  does not require chunking
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i64, ov::Shape{2, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i64, ov::Shape{2, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    const size_t tested_max_allowed_chunks = 1;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        tested_max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("exceeding max_allowed_chunks before chunking limit: 1; actual: 2"));
}

TEST_F(RerankChunkingTest, MaxAllowedChunkExceededAfterChunkingTest) {
    std::vector<int64_t> input_ids_data = {
        101, 102, 1,    // 2 tokens
        101, 102, 103,  // 3 tokens
    };
    std::vector<int64_t> attention_mask_data = {
        1, 1, 0,  // 2 tokens
        1, 1, 1,  // 3 tokens
    };
    size_t max_tokens_per_chunk = 2;  // makes the total chunks to be 3
    int64_t pad_token = 1;

    ov::Tensor in_input_ids(ov::element::i64, ov::Shape{2, 3}, input_ids_data.data());
    ov::Tensor in_attention_mask(ov::element::i64, ov::Shape{2, 3}, attention_mask_data.data());

    ov::Tensor out_input_ids, out_attention_mask;

    std::vector<size_t> chunk_mapping;
    const size_t tested_max_allowed_chunks = 2;
    auto status = chunkDocuments(
        in_input_ids, in_attention_mask,
        out_input_ids, out_attention_mask,
        chunk_mapping, max_tokens_per_chunk,
        tested_max_allowed_chunks, pad_token);

    ASSERT_EQ(status, absl::InvalidArgumentError("exceeding max_allowed_chunks after chunking limit: 2; actual: 3"));
}
