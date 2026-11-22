add_library(taw STATIC
  buf.c
  crc32c.c
  predictor.c
  rc.c
  version.c
)
target_include_directories(taw
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
