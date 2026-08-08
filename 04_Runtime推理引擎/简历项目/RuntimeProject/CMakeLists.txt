cmake_minimum_required(VERSION 3.16)
project(runtime_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(runtime_demo src/runtime_demo.cpp)
target_include_directories(runtime_demo PRIVATE include)
