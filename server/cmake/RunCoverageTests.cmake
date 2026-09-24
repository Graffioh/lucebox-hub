foreach(_coverage_required_var IN ITEMS
        CTEST_EXECUTABLE
        BUILD_DIR
        PROFILE_DIR
        CTEST_EXCLUDE_REGEX)
    if(NOT DEFINED ${_coverage_required_var})
        message(FATAL_ERROR "${_coverage_required_var} is required")
    endif()
endforeach()
unset(_coverage_required_var)

set(ENV{LLVM_PROFILE_FILE} "${PROFILE_DIR}/%m_%p.profraw")
set(_coverage_ctest_args --test-dir "${BUILD_DIR}" --output-on-failure)
if(NOT "${CTEST_EXCLUDE_REGEX}" STREQUAL "")
    list(APPEND _coverage_ctest_args -E "${CTEST_EXCLUDE_REGEX}")
endif()
execute_process(
    COMMAND "${CTEST_EXECUTABLE}" ${_coverage_ctest_args}
    RESULT_VARIABLE _coverage_ctest_result)
unset(_coverage_ctest_args)

include("${CMAKE_CURRENT_LIST_DIR}/GenerateCoverageReport.cmake")

if(NOT _coverage_ctest_result EQUAL 0)
    message(FATAL_ERROR "CTest failed with exit code ${_coverage_ctest_result}")
endif()
