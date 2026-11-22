# CLI roundtrip: cmake -DTAW=<taw> -DSOURCES=<f1|f2|...> -DWORK=<dir> -P cli_roundtrip.cmake
file(REMOVE_RECURSE "${WORK}")
file(MAKE_DIRECTORY "${WORK}")

# Input: the given files concatenated three times (tens of KiB of text).
set(INPUT "${WORK}/input.bin")
file(WRITE "${INPUT}" "")
foreach(rep 1 2 3)
  string(REPLACE "|" ";" files "${SOURCES}")
  foreach(f IN LISTS files)
    file(READ "${f}" chunk)
    file(APPEND "${INPUT}" "${chunk}")
  endforeach()
endforeach()

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "failed (${rc}): ${ARGN}\n${err}")
  endif()
endfunction()

function(same a b)
  execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files "${a}" "${b}"
                  RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "files differ: ${a} ${b}")
  endif()
endfunction()

# Stream variant.
run("${TAW}" c -l 3 "${INPUT}" "${WORK}/in.aw")
run("${TAW}" d "${WORK}/in.aw" "${WORK}/out.aw.bin")
same("${INPUT}" "${WORK}/out.aw.bin")

# Seekable variant, selected by extension, small blocks.
run("${TAW}" c -B 4096 "${INPUT}" "${WORK}/in.awseek")
run("${TAW}" d "${WORK}/in.awseek" "${WORK}/out.seek.bin")
same("${INPUT}" "${WORK}/out.seek.bin")

execute_process(COMMAND "${TAW}" i "${WORK}/in.awseek" OUTPUT_VARIABLE info)
if(NOT info MATCHES "kind: +awseek" OR NOT info MATCHES "block_size: +4096")
  message(FATAL_ERROR "unexpected info output:\n${info}")
endif()

# Range extraction straddling a block boundary.
file(SIZE "${INPUT}" size)
if(size LESS 9000)
  message(FATAL_ERROR "INPUT too small for the range test")
endif()
run("${TAW}" x "${WORK}/in.awseek" 4000 5000 "${WORK}/range.bin")
file(READ "${INPUT}" want OFFSET 4000 LIMIT 5000 HEX)
file(READ "${WORK}/range.bin" got HEX)
if(NOT want STREQUAL got)
  message(FATAL_ERROR "range extract mismatch")
endif()

# Out-of-range extraction must fail.
execute_process(COMMAND "${TAW}" x "${WORK}/in.awseek" ${size} 1 "${WORK}/bad.bin"
                RESULT_VARIABLE rc ERROR_QUIET)
if(rc EQUAL 0)
  message(FATAL_ERROR "out-of-range extract succeeded")
endif()
