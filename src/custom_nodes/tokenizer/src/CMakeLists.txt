cmake_minimum_required(VERSION 3.10)
project(tokenizer)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(BLINGFIRE_INSTALL_DIR ${CMAKE_BINARY_DIR}/external/blingfire)

ExternalProject_Add(blingfire
    PREFIX blingfire
    GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
    GIT_TAG ${BLINGFIRE_SHA}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BLINGFIRE_INSTALL_DIR}
)

set(BLINGFIRE_STATIC_LIB_DIR ${CMAKE_BINARY_DIR}/src/blingfire/src/blingfire-build)
set(BLINGFIRE_STATIC_LIBS
    ${BLINGFIRE_STATIC_LIB_DIR}/libbingfirtinydll_static.a
    ${BLINGFIRE_STATIC_LIB_DIR}/libfsaClient.a)

set(UTIL_DIRS
    ../../common/       # for common/utils.hpp
    ../../../)          # for custom_node_interface.h

add_library(tokenizer SHARED model.cpp tokenizer.cpp)
target_include_directories(tokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include ${UTIL_DIRS})
target_link_libraries(tokenizer ${BLINGFIRE_STATIC_LIBS})
add_dependencies(tokenizer blingfire)

add_library(detokenizer SHARED model.cpp detokenizer.cpp)
target_include_directories(detokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include ${UTIL_DIRS})
target_link_libraries(detokenizer ${BLINGFIRE_STATIC_LIBS})
add_dependencies(detokenizer blingfire)
