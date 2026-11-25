cmake_minimum_required(VERSION 3.20)
project(xray_enhance_gl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

add_library(xenh
  src/image.cpp
  src/gl.cpp
  src/lut.cpp
  src/enhance_cpu.cpp
  src/enhance_gpu.cpp
  src/metrics.cpp)
target_include_directories(xenh PUBLIC src)
target_link_libraries(xenh PUBLIC OpenGL::OpenGL OpenGL::EGL)
target_compile_definitions(xenh PRIVATE XENH_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
# Headless only: stop the EGL headers from pulling in X11.
target_compile_definitions(xenh PUBLIC EGL_NO_X11 MESA_EGL_NO_X11_HEADERS)
# -ffp-contract=off keeps the CPU reference from fusing multiply-add, so it rounds like the plain formulas.
target_compile_options(xenh PUBLIC -Wall -Wextra -ffp-contract=off)

add_executable(unit_tests tests/test_main.cpp)
target_link_libraries(unit_tests PRIVATE xenh)

add_executable(enhance apps/enhance.cpp)
target_link_libraries(enhance PRIVATE xenh)

add_executable(study apps/study.cpp)
target_link_libraries(study PRIVATE xenh)

enable_testing()
add_test(NAME unit COMMAND unit_tests)
