# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.25)
project(tadamw VERSION 0.1.0 LANGUAGES C
        DESCRIPTION "Archives whose predictor is a small model fitted with AdamW")

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# libtadamw: either half can be left out. The tools build the commands their
# half provides; the tests need both.
option(TADAMW_ENCODER "Build the encoder (compression, model fitting)" ON)
option(TADAMW_DECODER "Build the decoder (decompression, random access)" ON)
option(TADAMW_TOOLS "Build the taw and taw-nx command-line tools" ON)
option(TADAMW_TESTS "Build the tests (needs the encoder and the decoder)" ON)
option(TADAMW_INSTALL "Generate install rules (library, headers, CMake package, pkg-config)" ON)
option(BUILD_SHARED_LIBS "Build libtadamw as a shared library" OFF)
option(TADAMW_WERROR "Treat warnings as errors" OFF)
option(TADAMW_SANITIZE "Build with ASan+UBSan" OFF)
# GPU fitting needs the Vulkan headers and glslangValidator at build time
# only: the loader is opened at run time (dlopen), if it is there at all.
find_path(TADAMW_VULKAN_INCLUDE_DIR vulkan/vulkan.h)
find_program(TADAMW_GLSLANG glslangValidator)
if(TADAMW_VULKAN_INCLUDE_DIR AND TADAMW_GLSLANG)
  set(tadamw_gpu_default ON)
else()
  set(tadamw_gpu_default OFF)
endif()
option(TADAMW_GPU "Encoder: fit models on a GPU (Vulkan, loaded at run time)"
       ${tadamw_gpu_default})
if(NOT TADAMW_ENCODER AND NOT TADAMW_DECODER)
  message(FATAL_ERROR "TADAMW_ENCODER and TADAMW_DECODER are both OFF: nothing to build")
endif()

include(GNUInstallDirs)

# Probabilities must be bit-identical between encoder and decoder, so floating
# point contraction (FMA fusion) and fast-math are never allowed. See
# docs/design.md, "Determinism".
add_compile_options(-ffp-contract=off -fno-fast-math)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
                      -Wmissing-prototypes -Wconversion -Wno-sign-conversion)
  if(TADAMW_WERROR)
    add_compile_options(-Werror)
  endif()
  if(TADAMW_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

add_subdirectory(src)
if(TADAMW_TOOLS)
  add_subdirectory(tools)
endif()
if(TADAMW_TESTS AND TADAMW_ENCODER AND TADAMW_DECODER)
  enable_testing()
  add_subdirectory(tests)
elseif(TADAMW_TESTS)
  message(STATUS "tadamw: tests skipped (they need the encoder and the decoder)")
endif()
