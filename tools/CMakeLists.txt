add_executable(taw_cli taw.c)
set_target_properties(taw_cli PROPERTIES OUTPUT_NAME taw)
target_link_libraries(taw_cli PRIVATE taw)
