cmake_minimum_required(VERSION 2.8)
project(test)



####Specify the path of ffdv/net
set(DLPC_ROOT_DIR /home/sherry/dlpc)



set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(${PROJECT_SOURCE_DIR} ${DLPC_ROOT_DIR})
link_directories(${DLPC_ROOT_DIR}/lib)


set(src test.cpp)

add_executable(test_dbn ${src})
target_link_libraries(test_dbn dlpc)

