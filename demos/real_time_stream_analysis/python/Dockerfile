#
# Copyright (c) 2022 Intel Corporation
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

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt -y update && apt install -y libopencv-dev python3-opencv python3-pip
COPY requirements.txt /real_time_stream_analysis/
RUN pip3 install -r /real_time_stream_analysis/requirements.txt
COPY templates /real_time_stream_analysis/templates
COPY use_cases /real_time_stream_analysis/use_cases
COPY *.py /real_time_stream_analysis/
WORKDIR /real_time_stream_analysis

ENTRYPOINT ["python3", "real_time_stream_analysis.py"]