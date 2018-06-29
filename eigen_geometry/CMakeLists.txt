cmake_minimum_required(VERSION 2.8)

project (eigen)

include_directories(/usr/local/include/eigen3)

add_executable(eigen src/test.cpp)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

