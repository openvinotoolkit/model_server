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
#pragma once

namespace ovms {
/**
 * @brief Interface for periodic cleanup of resources that are no longer in use.
 *
 * Implemented by ModelManager to release shared resources (e.g. custom node library wrappers)
 * that have been retired but may still be referenced by in-flight requests. A background cleaner
 * thread calls cleanupResources() at regular intervals to collect resources whose reference
 * counts have dropped to the sole owner.
 */
class ResourcesCleaner {
public:
    virtual ~ResourcesCleaner() = default;
    virtual void cleanupResources() = 0;
};
}  // namespace ovms
