#!/bin/bash
#
# Copyright (c) 2026 Intel Corporation
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

# Builds GStreamer from source into ${GST_PREFIX}. Single source of truth for the
# plugin set, shared by Dockerfile.ubuntu (build container) and extras/gstreamer/Dockerfile.ubuntu.
# Legal: gpl=disabled, ugly=disabled - no GPL plugins are built.

set -euxo pipefail

GST_VERSION="${GST_VERSION:-1.26.2}"
GST_PREFIX="${GST_PREFIX:-/opt/gstreamer}"
JOBS="${JOBS:-8}"

export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    pkg-config \
    python3-pip \
    python3-venv \
    flex \
    bison \
    libglib2.0-dev \
    libmount-dev \
    libpcre2-dev \
    libffi-dev \
    libssl-dev \
    libdrm-dev \
    libva-dev \
    nasm \
    yasm \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    git
apt-get clean && rm -rf /var/lib/apt/lists/*

python3 -m venv /build_venv
/build_venv/bin/pip install --no-cache-dir meson==1.4.1 ninja==1.11.1.1
export PATH="/build_venv/bin:${PATH}"

mkdir -p /src && cd /src
git clone --depth 1 --branch "${GST_VERSION}" https://gitlab.freedesktop.org/gstreamer/gstreamer.git
cd /src/gstreamer

meson setup \
    -Dexamples=disabled \
    -Dtests=disabled \
    -Dgst-examples=disabled \
    -Ddevtools=disabled \
    -Dorc=disabled \
    -Dgpl=disabled \
    -Dpython=disabled \
    -Dlibnice=disabled \
    -Dugly=disabled \
    -Dgst-plugins-bad:gpl=disabled \
    -Dgst-plugins-bad:va=enabled \
    -Dgst-plugins-bad:doc=disabled \
    -Dgst-plugins-bad:nls=disabled \
    -Dgst-plugins-bad:directfb=disabled \
    -Dgst-plugins-bad:openni2=disabled \
    -Dgst-plugins-bad:fdkaac=disabled \
    -Dgst-plugins-bad:ladspa=disabled \
    -Dgst-plugins-bad:assrender=disabled \
    -Dgst-plugins-bad:bs2b=disabled \
    -Dgst-plugins-bad:flite=disabled \
    -Dgst-plugins-bad:rtmp=disabled \
    -Dgst-plugins-bad:opencv=disabled \
    -Dgst-plugins-bad:sbc=disabled \
    -Dgst-plugins-bad:teletext=disabled \
    -Dgst-plugins-bad:x265=disabled \
    -Dgst-plugins-bad:webrtcdsp=disabled \
    -Dgst-plugins-bad:dash=disabled \
    -Dgst-plugins-bad:openjpeg=disabled \
    -Dgst-plugins-bad:soundtouch=disabled \
    -Dgst-plugins-bad:isac=disabled \
    -Dgst-plugins-base:nls=disabled \
    -Dgst-plugins-base:gl=disabled \
    -Dgst-plugins-base:pango=disabled \
    -Dgst-plugins-base:xvideo=disabled \
    -Dgst-plugins-good:nls=disabled \
    -Dgst-plugins-good:libcaca=disabled \
    -Dgst-plugins-good:lame=disabled \
    -Dgst-plugins-good:flac=disabled \
    -Dgst-plugins-good:dv=disabled \
    -Dgst-plugins-good:adaptivedemux2=disabled \
    --buildtype=release \
    --prefix="${GST_PREFIX}" \
    --libdir=lib \
    --libexecdir=bin \
    build/

ninja -C build/ -j"${JOBS}"
meson install -C build/

# glib headers are a build dependency; copy them so the prefix is self-contained for Bazel.
cp -r /usr/include/glib-2.0 "${GST_PREFIX}/include/glib-2.0"
cp -r /usr/lib/x86_64-linux-gnu/glib-2.0/include/. "${GST_PREFIX}/include/glib-2.0/"
