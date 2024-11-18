#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# The tf.patch "tf logging macros" part is changing the logging.h file because it has macro redefinition conflicts
# with mediapipe logging.h on the following macros and functions connected with them.
# We are adding glog/logging.h include and the github glog dependency in the tensorflow project
# so that it uses the same macros from common dependency and we remove the conflicting macros from original file.

# define LOG(severity) _TF_LOG_##severit
# define VLOG_IS_ON(lvl)
# define VLOG(level)
# define DVLOG(verbose_level) 
# define LOG_EVERY_N(severity, n)  
# define LOG_FIRST_N(severity, n)    
# define CHECK(condition)
# define CHECK_OP_LOG(name, op, val1, val2)
# define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2) 
# define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)  
# define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
# define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
# define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
# define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
# define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)
  
# define DCHECK_EQ(x, y) _TF_DCHECK_NOP(x, y)
# define DCHECK_NE(x, y) _TF_DCHECK_NOP(x, y)
# define DCHECK_LE(x, y) _TF_DCHECK_NOP(x, y)
# define DCHECK_LT(x, y) _TF_DCHECK_NOP(x, y)
# define DCHECK_GE(x, y) _TF_DCHECK_NOP(x, y)

licenses(["notice"])

exports_files([
    "listen.patch",
    "tf.patch",
    "net_http.patch",
])