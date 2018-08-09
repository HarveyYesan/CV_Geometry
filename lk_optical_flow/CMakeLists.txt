cmake_minimum_required( VERSION 2.8)

project(VO2)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LK src/lk.cpp)
target_link_libraries(LK ${OpenCV_LIBS})


