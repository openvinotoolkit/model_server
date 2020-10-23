#
# Copyright (c) 2020 Intel Corporation
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

OUT_LIB = libsampleloader.so

LIB_INC = -Iinclude -I../../ -Irapidjson-1.1.0/include
CFLAGS = -fPIC -Wl,-rpath,. -fpic 
LDFLAGS = -ldl -lpthread

all:
	wget https://github.com/Tencent/rapidjson/archive/v1.1.0.zip
	unzip v1.1.0.zip

	g++ -c -g -fPIC $(LIB_INC) *.cpp 
	g++ *.o -g $(CFLAGS) $(LDFLAGS) $(LIB_INC) -shared -o $(OUT_LIB) 
	$(RM) *.o
	$(RM) v1.1.0.zip
	$(RM) rapidjson-1.1.0 -r

clean:
	$(RM) $(OUT_LIB) *.o
	$(RM) v1.1.0.zip
	$(RM) rapidjson-1.1.0 -r
