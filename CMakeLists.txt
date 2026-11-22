cmake_minimum_required(VERSION 3.25)
project(tar_aw VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(TAW_WERROR "Treat warnings as errors" OFF)
option(TAW_SANITIZE "Build with ASan+UBSan" OFF)

# Probabilities must be bit-identical between encoder and decoder, so floating
# point contraction (FMA fusion) and fast-math are never allowed. See
# docs/design.md, "Determinism".
add_compile_options(-ffp-contract=off -fno-fast-math)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
                      -Wmissing-prototypes -Wconversion -Wno-sign-conversion)
  if(TAW_WERROR)
    add_compile_options(-Werror)
  endif()
  if(TAW_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(tests)
