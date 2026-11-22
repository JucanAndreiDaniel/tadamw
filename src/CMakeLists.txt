# SPDX-License-Identifier: GPL-3.0-or-later
add_library(taw STATIC
  api.c
  aw.c
  awseek.c
  buf.c
  crc32c.c
  fit.c
  header.c
  p1_ctx.c
  p1_net.c
  predictor.c
  rc.c
  version.c
)
target_include_directories(taw
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# x86 SIMD kernels are compiled with their own ISA flags and picked at run
# time by CPUID; everything else stays baseline.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  target_sources(taw PRIVATE p1_avx2.c p1_avx512.c)
  set_source_files_properties(p1_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c")
  set_source_files_properties(p1_avx512.c PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mf16c")
  target_compile_definitions(taw PRIVATE TAW_X86_SIMD=1)
else()
  target_compile_definitions(taw PRIVATE TAW_X86_SIMD=0)
endif()
