//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "modelversion.hpp"

namespace ovms {

/**
 * @brief Base class for model version policy types
 */
class ModelVersionPolicy {
protected:
    ModelVersionPolicy() {}
    virtual ~ModelVersionPolicy() {}

public:
    /**
     * @brief Strips out model versions list passed by parameter depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    virtual model_versions_t filter(model_versions_t versions) const = 0;

    /**
     * @brief Creates default model version policy, by default only one version (highest) should be served
     * 
     * @param highestVersion highest version
     * @return default version policy
     */
    static std::shared_ptr<ModelVersionPolicy> getDefaultVersionPolicy();

    /**
     * @brief Converts ModelVersionPolicy to readable string
     */
    virtual operator std::string() const = 0;
};

/**
 * @brief Model version policy that enables all available versions
 */
class AllModelVersionPolicy : public ModelVersionPolicy {
public:
    /**
     * @brief Default constructor, nothing needs to be specified since all versions will be served
     */
    AllModelVersionPolicy() {}

    /**
     * @brief Filters passed versions depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    model_versions_t filter(model_versions_t versions) const override {
        return versions;
    }

    operator std::string() const override;
};

/**
 * @brief Model version policy for explicitly specifying which versions should be enabled
 */
class SpecificModelVersionPolicy : public ModelVersionPolicy {
    model_versions_t specificVersions;

public:
    /**
     * @brief Default constructor
     * 
     * @param versions list of all model versions that should be served
     */
    SpecificModelVersionPolicy(const model_versions_t& versions) :
        specificVersions(versions) {
        std::sort(specificVersions.begin(), specificVersions.end());
    }

    /**
     * @brief Filters passed versions depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    model_versions_t filter(model_versions_t versions) const override;

    operator std::string() const override;
};

/**
 * @brief Model version policy for serving only X latest versions
 */
class LatestModelVersionPolicy : public ModelVersionPolicy {
    size_t numVersions;

public:
    /**
     * @brief Default constructor
     * 
     * @param numVersions number of latest versions to be served
     */
    LatestModelVersionPolicy(size_t numVersions = 1) :
        numVersions(numVersions) {}

    /**
     * @brief Filters passed versions depending on internal state
     * 
     * @param versions model versions to filter
     * @return Filtered version list
     */
    model_versions_t filter(model_versions_t versions) const override;

    operator std::string() const override;
};

}  //  namespace ovms
