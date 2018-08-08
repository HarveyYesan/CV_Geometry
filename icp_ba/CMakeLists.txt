cmake_minimum_required( VERSION 2.8)

project(VO1)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3.0 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR}
                    "/usr/local/include/eigen3")

add_executable(icp src/pose_estimation_3d3d.cpp)
target_link_libraries(icp ${OpenCV_LIBS}
                          ${CSPARSE_LIBRARY}
                          g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)




