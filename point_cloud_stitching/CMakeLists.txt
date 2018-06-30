cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

project(map)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(Eigen REQUIRED)
include_directories(/usr/local/include/eigen3)

find_package(PCL REQUIRED)

#list (REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

#include_directories("/usr/local/pcl/visualization/include/pcl")
include_directories(/usr/local/include/pcl-1.8/)
include_directories(/usr/local/pcl/visualization/include/)
add_definitions(${PCL_DEFINITIONS})

add_executable(joinMap src/joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES})


