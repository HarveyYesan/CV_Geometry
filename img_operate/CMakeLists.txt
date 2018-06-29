cmake_minimum_required( VERSION 2.8 )
project( img_operate )

# add c++ 11
set( CMAKE_CXX_FLAGS "-std=c++11" )

# look for opencv lib
find_package( OpenCV REQUIRED )
# add opencv header
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( img_operate img_operate.cpp )
# link opencv lib
target_link_libraries( img_operate ${OpenCV_LIBS} )
