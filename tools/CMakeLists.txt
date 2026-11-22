# SPDX-License-Identifier: GPL-3.0-or-later
# The tools use the public API only (tadamw::tadamw) and build the commands
# of the halves that are enabled.
add_executable(taw_cli taw.c)
set_target_properties(taw_cli PROPERTIES OUTPUT_NAME taw)
target_link_libraries(taw_cli PRIVATE tadamw::tadamw)

if(TADAMW_INSTALL)
  install(TARGETS taw_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
