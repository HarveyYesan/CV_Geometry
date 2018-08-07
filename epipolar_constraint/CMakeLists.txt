cmake_minimum_required( VERSION 2.8)
project(VO1)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(OpenCV 3.3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(epipolar_constraint src/pose_estimation_2d2d.cpp)
target_link_libraries(epipolar_constraint ${OpenCV_LIBS})

add_executable(triangulation src/triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

