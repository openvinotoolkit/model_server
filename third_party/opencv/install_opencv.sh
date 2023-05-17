#!/bin/bash
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
set -exo pipefail
#===================================================================================================
# Option parsing

os=${os:-auto}
opencv_branch=${opencv_branch:-4.7.0}
work_dir=${work_dir:-/opt}


#===================================================================================================
# OS detection

if [ "$os" == "auto" ] ; then
    os=$( . /etc/os-release ; echo "${ID}${VERSION_ID}" )
    if [[ "$os" =~ "rhel8".* ]] ; then
      os="rhel8"
    fi
    case $os in
        rhel8|ubuntu18.04|ubuntu20.04|ubuntu21.10|ubuntu22.04) [ -z "$print" ] && echo "Detected OS: ${os}" ;;
        *) echo "Unsupported OS: ${os:-detection failed}" >&2 ; exit 1 ;;
    esac
fi

#===================================================================================================
# OpenCV installation

if [ "$os" == "ubuntu20.04" ] || [ "$os" == "ubuntu22.04" ] ; then
    export DEBIAN_FRONTEND=noninteractive
    apt update && apt install -y build-essential git cmake \
        && rm -rf /var/lib/apt/lists/*
elif [ "$os" == "rhel8" ] ; then
    yum install -d6 -y git cmake gcc-c++
else
    echo "Internal script error: unsupported OS" >&2
    exit 3
fi

current_working_dir=$(pwd)

cd $work_dir
rm -rf $opencv_branch $work_dir/opencv_repo
rm -rf $opencv_branch $work_dir/opencv_contrib_repo
git clone https://github.com/opencv/opencv.git --depth 1 -b $opencv_branch $work_dir/opencv_repo
git clone https://github.com/opencv/opencv_contrib.git --depth 1 -b $opencv_branch $work_dir/opencv_contrib_repo
cd $work_dir/opencv_repo
mkdir -p $work_dir/opencv_repo/build
cd $work_dir/opencv_repo/build
cmake $(cat $current_working_dir/opencv_cmake_flags.txt) $work_dir/opencv_repo && \
    make "-j$(nproc)" && \
    make install

#===================================================================================================
# end

