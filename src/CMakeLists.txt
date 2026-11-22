add_library(taw STATIC
  version.c
)
target_include_directories(taw
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
