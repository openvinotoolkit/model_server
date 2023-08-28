#!/bin/bash
cd flatbuffers && mkdir build && cd build && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) && cd ../..