cmake_minimum_required(VERSION 3.20)
project(bokeh_from_depth)

include_directories ("D:\\tbb\\include")

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH "D:\\tbb")
add_executable(bokeh_from_depth main.cpp)

set(OpenCV_DIR H:\\opencv-4.5.3-build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(bokeh_from_depth ${OpenCV_LIBS})

find_package(TBB REQUIRED)
target_link_libraries(bokeh_from_depth "D:\\tbb\\lib\\intel64\\vc14\\tbb.lib")


#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()