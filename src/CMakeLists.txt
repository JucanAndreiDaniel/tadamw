add_library(taw STATIC
  api.c
  aw.c
  awseek.c
  buf.c
  crc32c.c
  header.c
  predictor.c
  rc.c
  version.c
)
target_include_directories(taw
  PUBLIC  ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
