cmake_minimum_required ( VERSION 2.8)

project(VO2)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3.0 REQUIRED)
find_package(G2O REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    "/usr/local/include/eigen3")

add_executable(sparse_direct src/dierct_sparse.cpp)
target_link_libraries(sparse_direct ${OpenCV_LIBS}
                                     g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension)

add_executable(semidense_direct src/direct_semidense.cpp)
target_link_libraries(semidense_direct ${OpenCV_LIBS}
                                     g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension)



