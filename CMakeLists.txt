# cmake_minimum_required(VERSION <specify CMake version here>)
project(FFT_quantization)

set(CMAKE_CXX_STANDARD 14)
cmake_minimum_required(VERSION 3.10)
set(PYTHON_LIBRARY python3.6m)
set(PYTHON_INCLUDE_DIR /usr/include/python3.6 /home/user/.local/lib/python3.6/site-packages/numpy/core/include/)

find_package(xtensor REQUIRED)
find_package(PythonLibs 3.6)

#find_library(ARMADILLO_LIB armadillo)
#if(NOT ARMADILLO_LIB)
#    message(FATAL_ERROR "armadillo library not found")
#endif()

add_executable(FFT_quantization main.cpp)
target_include_directories(FFT_quantization PUBLIC third-party/xtensor/include PRIVATE ${PYTHON_INCLUDE_DIRS})
# armadillo
target_link_libraries(FFT_quantization xtensor ${PYTHON_LIBRARIES})
