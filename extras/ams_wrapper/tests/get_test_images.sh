#!/bin/bash

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

# Script for getting some test images with proper Creative Commons licenses
mkdir -p test_images

if [[ ! -f test_images/2207159142_8206ab6984.jpg ]]; then
wget https://live.staticflickr.com/2106/2207159142_8206ab6984.jpg && mv 2207159142_8206ab6984.jpg test_images/
fi

if [[ ! -f test_images/Khao_Manee_cat_in_Tamra_cat_Rattanakosin.png ]]; then
wget https://upload.wikimedia.org/wikipedia/commons/e/e0/Khao_Manee_cat_in_Tamra_cat_Rattanakosin.png && mv Khao_Manee_cat_in_Tamra_cat_Rattanakosin.png test_images/
fi

if [[ ! -f test_images/sails.bmp ]]; then
wget https://homepages.cae.wisc.edu/~ece533/images/sails.bmp && mv sails.bmp test_images/ 
fi