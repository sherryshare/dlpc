cmake_minimum_required(VERSION 2.8)
MESSAGE(STATUS "This is SOURCE dir" ${PROJECT_SOURCE_DIR})
PROJECT(test)

add_definitions(-g)


set(DLPC_ROOT_DIR /home/sherry/dlpc)



set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(${PROJECT_SOURCE_DIR} ${DLPC_ROOT_DIR})
link_directories(${DLPC_ROOT_DIR}/lib)


set(src testDBN.cpp)
#set(src testRBM.cpp)
#set(src testdA.cpp)
#set(src testSdA.cpp)
#set(src testLR.cpp)

add_executable(test ${src})
target_link_libraries(test dlpc)

