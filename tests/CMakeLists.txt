# Each test_<name>.c is a standalone executable registered with CTest.
function(taw_test name)
  add_executable(test_${name} test_${name}.c)
  target_link_libraries(test_${name} PRIVATE taw)
  target_include_directories(test_${name} PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                  ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND test_${name})
endfunction()

taw_test(version)
taw_test(crc32c)
taw_test(rc)
taw_test(aw)
taw_test(awseek)

add_test(NAME cli_version COMMAND taw_cli --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "^taw [0-9]+\\.[0-9]+\\.[0-9]+")

add_test(NAME cli_roundtrip
         COMMAND ${CMAKE_COMMAND} -DTAW=$<TARGET_FILE:taw_cli>
                 "-DSOURCES=${PROJECT_SOURCE_DIR}/docs/format.md|${PROJECT_SOURCE_DIR}/docs/design.md"
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli_work
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cli_roundtrip.cmake)
