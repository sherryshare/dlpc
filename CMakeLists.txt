cmake_minimum_required(VERSION 2.8)
project(dlpc)
add_definitions(-std=c++11)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(${PROJECT_SOURCE_DIR})
link_directories(${PROJECT_SOURCE_DIR}/lib)


set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


set(src DBN.cpp
	RBM.cpp
	LogisticRegression.cpp
	HiddenLayer.cpp
	SdA.cpp
	dA.cpp
)

add_library(dlpc_dynamic SHARED ${src})
add_library(dlpc_static STATIC ${src})




set_target_properties(dlpc_dynamic PROPERTIES OUTPUT_NAME "dlpc")
set_target_properties(dlpc_static PROPERTIES OUTPUT_NAME "dlpc")
set_target_properties(dlpc_dynamic PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(dlpc_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
