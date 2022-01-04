cmake_minimum_required(VERSION 3.20)
project(bokeh_from_depth)
project(bokeh_from_depth_lib)

include_directories ("D:\\tbb\\include")

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH "D:\\tbb")
add_executable(bokeh_from_depth main.cpp bokeh_utils.cpp bokeh_utils.h)
add_library(bokeh_from_depth_lib SHARED bokeh_utils.h bokeh_utils.cpp)

set(OpenCV_DIR H:\\opencv-4.5.3-build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(bokeh_from_depth ${OpenCV_LIBS})
#target_link_libraries(bokeh_from_depth_lib ${OpenCV_LIBS})

find_package(TBB REQUIRED)
#target_link_libraries(bokeh_from_depth "D:\\tbb\\lib\\intel64\\vc14\\tbb.lib")
#target_link_libraries(bokeh_from_depth_lib "D:\\tbb\\lib\\intel64\\vc14\\tbb.lib")

set( CMAKE_PREFIX_PATH "H:\\ECE496\\libtorch" )
include_directories( "H:\\ECE496\\libtorch\\include" )
include_directories( "H:\\ECE496\\libtorch\\include\\torch\\csrc\\api\\include" )
find_package( Torch REQUIRED )

target_link_libraries(bokeh_from_depth "${TORCH_LIBRARIES}" "D:\\tbb\\lib\\intel64\\vc14\\tbb.lib" "${OpenCV_LIBS}")
target_link_libraries(bokeh_from_depth_lib "${TORCH_LIBRARIES}" "D:\\tbb\\lib\\intel64\\vc14\\tbb.lib" "${OpenCV_LIBS}")




#FIND_PACKAGE( OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()