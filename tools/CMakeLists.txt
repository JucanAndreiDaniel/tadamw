# SPDX-License-Identifier: GPL-3.0-or-later
# The tools use the public API only (tadamw::tadamw) and build the commands
# of the halves that are enabled.
add_executable(taw_cli taw.c)
set_target_properties(taw_cli PROPERTIES OUTPUT_NAME taw)
target_link_libraries(taw_cli PRIVATE tadamw::tadamw)

# taw-nx: XCI/NSP transcoder. It needs OpenSSL (libcrypto) and LZ4 for the
# hactool copy in hactool/ (SciresM's hactool, ISC, as ScarletNX carries it,
# GPL-3.0-or-later changes); without them it is skipped.
option(TADAMW_NX "Build the taw-nx XCI/NSP transcoder" ON)
find_package(OpenSSL 3 COMPONENTS Crypto)
find_path(LZ4_INCLUDE_DIR lz4.h)
find_library(LZ4_LIBRARY lz4)
if(TADAMW_NX AND OpenSSL_FOUND AND LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
  set(hactool_src ${CMAKE_CURRENT_SOURCE_DIR}/hactool)
  set(hactool_sources save sha aes extkeys rsa npdm bktr kip packages pki pfs0
      hfs0 nca0_romfs romfs utils nax0 nso nca xci filepath ConvertUTF storage
      snx_hac zbic_stub)
  list(TRANSFORM hactool_sources PREPEND "${hactool_src}/")
  list(TRANSFORM hactool_sources APPEND .c)
  add_library(hactool_lib STATIC ${hactool_sources})
  target_include_directories(hactool_lib PRIVATE ${hactool_src} ${LZ4_INCLUDE_DIR})
  target_compile_definitions(hactool_lib PRIVATE _BSD_SOURCE _POSIX_SOURCE
      _POSIX_C_SOURCE=200112L _DEFAULT_SOURCE _FILE_OFFSET_BITS=64)
  # hactool's own code is built as ScarletNX builds it: its warnings off, and
  # snx_io.h ahead of every file so that nothing it prints (keys among it)
  # reaches the console.
  target_compile_options(hactool_lib PRIVATE -w -include ${hactool_src}/snx_io.h)
  set_target_properties(hactool_lib PROPERTIES C_STANDARD 23 C_EXTENSIONS ON)
  target_link_libraries(hactool_lib PUBLIC OpenSSL::Crypto ${LZ4_LIBRARY})

  add_executable(taw_nx taw_nx.c)
  # gnu23: pread/pwrite, explicit_bzero and sysconf(_SC_NPROCESSORS_ONLN)
  set_target_properties(taw_nx PROPERTIES OUTPUT_NAME taw-nx C_EXTENSIONS ON)
  target_include_directories(taw_nx PRIVATE ${hactool_src})
  target_link_libraries(taw_nx PRIVATE tadamw::tadamw hactool_lib OpenSSL::Crypto)
  find_package(Threads REQUIRED)
  target_link_libraries(taw_nx PRIVATE Threads::Threads)
else()
  message(STATUS "taw-nx: skipped (needs TADAMW_NX, OpenSSL 3 libcrypto and LZ4)")
endif()

if(TADAMW_INSTALL)
  install(TARGETS taw_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  if(TARGET taw_nx)
    install(TARGETS taw_nx RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()
