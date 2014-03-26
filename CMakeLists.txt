cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

IF(WIN32 OR win64)
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll)
ELSE()
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()


find_package(PCL 1.7 REQUIRED)
find_package( OpenCV REQUIRED )
FIND_PACKAGE(CUDA REQUIRED)
 
SET(CUDA_NVCC_FLAGS "-arch=sm_20" CACHE STRING "nvcc flags" FORCE)
#SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

project(3DSceneClassification)

add_subdirectory(src)

include_directories(C:/opencv/build)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(C:/blepo/external64/Microsoft/Kinect)
include_directories(${3DSceneClassification_SOURCE_DIR}/include)
include_directories(../PCL_WindowsKinectSDK/include)

link_directories(${PCL_LIBRARY_DIRS})
link_directories(C:/blepo/external64/Microsoft/Kinect)
link_directories(../../PCL_WindowsKinectSDK/build/)

add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE 3DSceneClassification_HEADERS include/*.h)
file(GLOB_RECURSE 3DSceneClassification_SOURCES src/*.cpp)
set (3DSceneClassification_INCLUDE_DIRS "")
foreach (_headerFile ${3DSceneClassification_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND 3DSceneClassification_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES 3DSceneClassification_INCLUDE_DIRS)

include_directories(${3DSceneClassification_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(BuildGraph STATIC include/BuildGraph.h src/BuildGraph.cu)

add_executable (3DSceneClassification ${3DSceneClassification_SOURCES} ${3DSceneClassification_HEADERS})
target_link_libraries (3DSceneClassification BuildGraph ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${PCL_LIBRARIES} ${OpenCV_LIBS} Kinect10.lib debug Debug/PCL_WindowsKinectSDK optimized Release/PCL_WindowsKinectSDK)