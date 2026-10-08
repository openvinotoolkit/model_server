#!/usr/bin/env python3
# ****************************************************************************
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ****************************************************************************
"""Regenerates the real image fixtures used by decoded_image_size_test.cpp.

The C++ test asserts that tryEstimateDecodedImageSize() matches the decoded
size derived from these known dimensions. Keep WIDTH/HEIGHT in sync with the
kW / kH constants in the test if you change them.

Usage:  python generate_test_images.py
"""
import os

from PIL import Image

WIDTH = 7
HEIGHT = 11
HERE = os.path.dirname(os.path.abspath(__file__))


def path(name):
    return os.path.join(HERE, name)


def main():
    rgb = Image.new("RGB", (WIDTH, HEIGHT), (10, 20, 30))
    rgba = Image.new("RGBA", (WIDTH, HEIGHT), (10, 20, 30, 40))
    gray16 = Image.new("I;16", (WIDTH, HEIGHT), 1234)

    rgb.save(path("rgb8.png"), format="PNG")
    rgba.save(path("rgba8.png"), format="PNG")
    gray16.save(path("gray16.png"), format="PNG")
    rgb.save(path("rgb.jpg"), format="JPEG", quality=90)
    rgb.save(path("rgb24.bmp"), format="BMP")
    rgb.save(path("sample.gif"), format="GIF")
    rgb.save(path("lossy.webp"), format="WEBP", lossless=False, quality=80)
    rgb.save(path("lossless.webp"), format="WEBP", lossless=True)

    for f in sorted(os.listdir(HERE)):
        if f == os.path.basename(__file__):
            continue
        print(f"{f}: {os.path.getsize(path(f))} bytes")


if __name__ == "__main__":
    main()
