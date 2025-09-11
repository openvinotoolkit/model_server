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
#pragma once

#include "src/modelconfig.hpp"

#include "platform_utils.hpp"
#include "test_models.hpp"

const ovms::ModelConfig DUMMY_MODEL_CONFIG{
    "dummy",
    dummy_model_location,  // base path
    "CPU",                 // target device
    "1",                   // batchsize
    1,                     // NIREQ
    false,                 // is stateful
    true,                  // idle sequence cleanup enabled
    false,                 // low latency transformation enabled
    500,                   // stateful sequence max number
    "",                    // cache directory
    1,                     // model_version unused since version are read from path
    dummy_model_location,  // local path
};

const ovms::ModelConfig DUMMY_FP64_MODEL_CONFIG{
    "dummy_fp64",
    dummy_fp64_model_location,  // base path
    "CPU",                      // target device
    "1",                        // batchsize
    1,                          // NIREQ
    false,                      // is stateful
    true,                       // idle sequence cleanup enabled
    false,                      // low latency transformation enabled
    500,                        // stateful sequence max number
    "",                         // cache directory
    1,                          // model_version unused since version are read from path
    dummy_fp64_model_location,  // local path
};

const ovms::ModelConfig SUM_MODEL_CONFIG{
    "sum",
    sum_model_location,  // base path
    "CPU",               // target device
    "1",                 // batchsize
    1,                   // NIREQ
    false,               // is stateful
    true,                // idle sequence cleanup enabled
    false,               // low latency transformation enabled
    500,                 // stateful sequence max number
    "",                  // cache directory
    1,                   // model_version unused since version are read from path
    sum_model_location,  // local path
};

const ovms::ModelConfig INCREMENT_1x3x4x5_MODEL_CONFIG{
    "increment_1x3x4x5",
    increment_1x3x4x5_model_location,  // base path
    "CPU",                             // target device
    "1",                               // batchsize
    1,                                 // NIREQ
    false,                             // is stateful
    true,                              // idle sequence cleanup enabled
    false,                             // low latency transformation enabled
    500,                               // stateful sequence max number
    "",                                // cache directory
    1,                                 // model_version unused since version are read from path
    increment_1x3x4x5_model_location,  // local path
};

const ovms::ModelConfig PASSTHROUGH_MODEL_CONFIG{
    "passthrough",
    passthrough_model_location,  // base path
    "CPU",                       // target device
    "1",                         // batchsize
    1,                           // NIREQ
    false,                       // is stateful
    true,                        // idle sequence cleanup enabled
    false,                       // low latency transformation enabled
    500,                         // stateful sequence max number
    "",                          // cache directory
    1,                           // model_version unused since version are read from path
    passthrough_model_location,  // local path
};

const ovms::ModelConfig NATIVE_STRING_MODEL_CONFIG{
    "passthrough_string",
    passthrough_string_model_location,  // base path
    "CPU",                              // target device
    "",                                 // batchsize
    1,                                  // NIREQ
    false,                              // is stateful
    true,                               // idle sequence cleanup enabled
    false,                              // low latency transformation enabled
    500,                                // stateful sequence max number
    "",                                 // cache directory
    1,                                  // model_version unused since version are read from path
    passthrough_string_model_location,  // local path
};

const ovms::ModelConfig DUMMY_SAVED_MODEL_CONFIG{
    "dummy_saved_model",
    dummy_saved_model_location,  // base path
    "CPU",                       // target device
    "1",                         // batchsize
    1,                           // NIREQ
    false,                       // is stateful
    true,                        // idle sequence cleanup enabled
    false,                       // low latency transformation enabled
    500,                         // stateful sequence max number
    "",                          // cache directory
    1,                           // model_version unused since version are read from path
    dummy_saved_model_location,  // local path
};

const ovms::ModelConfig DUMMY_TFLITE_CONFIG{
    "dummy_tflite",
    dummy_tflite_location,  // base path
    "CPU",                  // target device
    "1",                    // batchsize
    1,                      // NIREQ
    false,                  // is stateful
    true,                   // idle sequence cleanup enabled
    false,                  // low latency transformation enabled
    500,                    // stateful sequence max number
    "",                     // cache directory
    1,                      // model_version unused since version are read from path
    dummy_tflite_location,  // local path
};

const ovms::ModelConfig SCALAR_MODEL_CONFIG{
    "scalar",
    scalar_model_location,  // base path
    "CPU",                  // target device
    "",                     // batchsize needs to be empty to emulate missing --batch_size param
    1,                      // NIREQ
    false,                  // is stateful
    true,                   // idle sequence cleanup enabled
    false,                  // low latency transformation enabled
    500,                    // stateful sequence max number
    "",                     // cache directory
    1,                      // model_version unused since version are read from path
    scalar_model_location,  // local path
};

const ovms::ModelConfig NO_NAME_MODEL_CONFIG{
    "no_name_output",
    no_name_output_model_location,  // base path
    "CPU",                          // target device
    "1",                            // batchsize
    1,                              // NIREQ
    false,                          // is stateful
    true,                           // idle sequence cleanup enabled
    false,                          // low latency transformation enabled
    500,                            // stateful sequence max number
    "",                             // cache directory
    1,                              // model_version unused since version are read from path
    no_name_output_model_location,  // local path
};

const ovms::Shape DUMMY_MODEL_SHAPE_META{1, 10};
