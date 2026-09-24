foreach(_coverage_required_var IN ITEMS
        LLVM_COV
        LLVM_PROFDATA
        PROFILE_DIR
        PROFILE_DATA
        OBJECT_LIST_FILE
        SOURCE_DIR
        HTML_DIR)
    if(NOT DEFINED ${_coverage_required_var} OR
       "${${_coverage_required_var}}" STREQUAL "")
        message(FATAL_ERROR "${_coverage_required_var} is required")
    endif()
endforeach()
unset(_coverage_required_var)

file(GLOB _coverage_raw_profiles "${PROFILE_DIR}/*.profraw")
if(NOT _coverage_raw_profiles)
    message(FATAL_ERROR
        "CTest produced no LLVM profiles in ${PROFILE_DIR}")
endif()

execute_process(
    COMMAND "${LLVM_PROFDATA}" merge -sparse ${_coverage_raw_profiles}
        -o "${PROFILE_DATA}"
    RESULT_VARIABLE _coverage_merge_result
)
if(NOT _coverage_merge_result EQUAL 0)
    message(FATAL_ERROR "llvm-profdata failed with exit code ${_coverage_merge_result}")
endif()

file(STRINGS "${OBJECT_LIST_FILE}" _coverage_objects)
if(NOT _coverage_objects)
    message(FATAL_ERROR "No coverage objects were generated in ${OBJECT_LIST_FILE}")
endif()

list(GET _coverage_objects 0 _coverage_main_object)
list(REMOVE_AT _coverage_objects 0)
set(_coverage_object_args)
foreach(_coverage_object IN LISTS _coverage_objects)
    list(APPEND _coverage_object_args -object "${_coverage_object}")
endforeach()
unset(_coverage_object)

execute_process(
    COMMAND "${LLVM_COV}" report "${_coverage_main_object}"
        ${_coverage_object_args}
        "-instr-profile=${PROFILE_DATA}"
        "${SOURCE_DIR}"
    RESULT_VARIABLE _coverage_report_result
)
if(NOT _coverage_report_result EQUAL 0)
    message(FATAL_ERROR "llvm-cov report failed with exit code ${_coverage_report_result}")
endif()

execute_process(
    COMMAND "${LLVM_COV}" show "${_coverage_main_object}"
        ${_coverage_object_args}
        "-instr-profile=${PROFILE_DATA}"
        -format=html
        "-output-dir=${HTML_DIR}"
        "${SOURCE_DIR}"
    RESULT_VARIABLE _coverage_html_result
)
if(NOT _coverage_html_result EQUAL 0)
    message(FATAL_ERROR "llvm-cov show failed with exit code ${_coverage_html_result}")
endif()

message(STATUS "Coverage summary: ${PROFILE_DATA}")
message(STATUS "Coverage HTML report: ${HTML_DIR}/index.html")
