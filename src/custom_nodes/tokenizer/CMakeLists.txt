cmake_minimum_required(VERSION 3.10)
project(node)

include(ExternalProject)

set(BLINGFIRE_SHA 5089d31914cbed7a24589e753bd6cd362a377fbb)

add_subdirectory(src)

if (WITH_TESTS)
    add_subdirectory(test)
endif()
