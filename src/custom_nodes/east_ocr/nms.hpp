//*****************************************************************************
// The MIT License (MIT)

// Copyright (c) 2015 Sergey Nuzhny

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//*****************************************************************************
#pragma once

#include <map>
#include <utility>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 6269 6294 6201)
#include "opencv2/opencv.hpp"
#pragma warning(pop)

/**
 * @brief nms
 * Non maximum suppression
 * @param srcRects
 * @param resRects
 * @param thresh
 * @param neighbors
 */
inline void nms(const std::vector<cv::Rect>& srcRects,
    std::vector<cv::Rect>& resRects,
    float thresh,
    int neighbors = 0) {
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        idxs.emplace(srcRects[i].br().y, i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs);) {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = static_cast<float>((rect1 & rect2).area());
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) {
                pos = idxs.erase(pos);
                ++neigborsCount;
            } else {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
            resRects.push_back(rect1);
    }
}

/**
 * @brief nms2
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param scores
 * @param resRects
 * @param thresh
 * @param neighbors
 */
template <typename T>
inline void nms2(const std::vector<cv::Rect>& srcRects,
    const std::vector<float>& scores,
    const std::vector<T>& metadata,
    std::vector<cv::Rect>& resRects,
    std::vector<float>& resScores,
    std::vector<T>& resMetadata,
    float thresh,
    int neighbors = 0,
    float minScoresSum = 0.f) {
    resRects.clear();
    resScores.clear();
    resMetadata.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    assert(srcRects.size() == scores.size());
    assert(srcRects.size() == metadata.size());

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        idxs.emplace(scores[i], i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];
        float score = scores[lastElem->second];
        const T& data = metadata[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs);) {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = static_cast<float>((rect1 & rect2).area());
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            } else {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum) {
            resRects.push_back(rect1);
            resScores.push_back(score);
            resMetadata.push_back(data);
        }
    }
}

///
enum class Methods {
    ClassicNMS,
    LinearNMS,
    GaussNMS
};

/**
 * @brief nms2
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param scores
 * @param resRects
 * @param thresh
 */
inline void soft_nms(const std::vector<cv::Rect>& srcRects,
    const std::vector<float>& scores,
    std::vector<cv::Rect>& resRects,
    std::vector<float>& resScores,
    float iou_thresh,
    float score_thresh,
    Methods method,
    float sigma = 0.5f) {
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    assert(srcRects.size() == scores.size());

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i) {
        if (scores[i] >= score_thresh)
            idxs.emplace(scores[i], i);
    }

    if (resRects.capacity() < idxs.size()) {
        resRects.reserve(idxs.size());
        resScores.reserve(idxs.size());
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0) {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        if (lastElem->first >= score_thresh) {
            resRects.push_back(rect1);
            resScores.push_back(lastElem->first);
        } else {
            break;
        }
        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs);) {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = static_cast<float>((rect1 & rect2).area());
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > iou_thresh) {
                float weight = 1.f;
                switch (method) {
                case Methods::ClassicNMS:
                    weight = 0;
                    break;
                case Methods::LinearNMS:
                    weight = 1.f - overlap;
                    break;
                case Methods::GaussNMS:
                    weight = exp(-(overlap * overlap) / sigma);
                    break;
                }

                float newScore = pos->first * weight;
                if (newScore < score_thresh) {
                    pos = idxs.erase(pos);
                } else {
                    auto n = idxs.extract(pos);
                    n.key() = newScore;
                    idxs.insert(std::move(n));
                    ++pos;
                }
            } else {
                ++pos;
            }
        }
    }
}
