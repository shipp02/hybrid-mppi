cmake_minimum_required(VERSION 3.18)
project(mppi_generated_acados_solver LANGUAGES C)

# Copy this file beside generated C, or point GENERATED_DIR at that directory.
set(GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Generated acados C directory")
set(ACADOS_ROOT "$ENV{ACADOS_SOURCE_DIR}" CACHE PATH "Target acados installation prefix")
if(NOT ACADOS_ROOT)
  message(FATAL_ERROR "Set -DACADOS_ROOT=/path/to/target/acados")
endif()
get_filename_component(GENERATED_DIR "${GENERATED_DIR}" ABSOLUTE)
get_filename_component(ACADOS_ROOT "${ACADOS_ROOT}" ABSOLUTE)

find_path(ACADOS_INCLUDE_DIR acados_c/ocp_nlp_interface.h
  PATHS "${ACADOS_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
foreach(dependency acados hpipm blasfeo)
  find_library(${dependency}_LIBRARY NAMES ${dependency}
    PATHS "${ACADOS_ROOT}/lib" "${ACADOS_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
  add_library(${dependency}_runtime UNKNOWN IMPORTED)
  set_target_properties(${dependency}_runtime PROPERTIES
    IMPORTED_LOCATION "${${dependency}_LIBRARY}")
endforeach()

# Model name includes acados' generated problem identifier; never hardcode it.
file(GLOB solver_sources CONFIGURE_DEPENDS "${GENERATED_DIR}/acados_solver_*.c")
list(LENGTH solver_sources solver_count)
if(NOT solver_count EQUAL 1)
  message(FATAL_ERROR "Expected one acados_solver_*.c in ${GENERATED_DIR}; found ${solver_count}")
endif()
list(GET solver_sources 0 solver_source)
get_filename_component(solver_name "${solver_source}" NAME_WE)
string(REGEX REPLACE "^acados_solver_" "" problem_name "${solver_name}")
file(GLOB generated_functions CONFIGURE_DEPENDS
  "${GENERATED_DIR}/*_model/*.c"
  "${GENERATED_DIR}/*_cost/*.c"
  "${GENERATED_DIR}/*_constraints/*.c")

add_library(tracker_solver SHARED ${solver_sources} ${generated_functions})
set_target_properties(tracker_solver PROPERTIES
  OUTPUT_NAME "acados_ocp_solver_${problem_name}"
  C_STANDARD 99 C_STANDARD_REQUIRED YES
  BUILD_RPATH "${ACADOS_ROOT}/lib;${ACADOS_ROOT}/lib64")
target_include_directories(tracker_solver PUBLIC
  "${GENERATED_DIR}"
  "${ACADOS_INCLUDE_DIR}"
  "${ACADOS_INCLUDE_DIR}/acados"
  "${ACADOS_INCLUDE_DIR}/blasfeo/include"
  "${ACADOS_INCLUDE_DIR}/hpipm/include")
target_link_libraries(tracker_solver PUBLIC acados_runtime hpipm_runtime blasfeo_runtime)
if(UNIX)
  target_link_libraries(tracker_solver PUBLIC m)
endif()

# Generated acados example exercises native solver without Python or renderer.
if(EXISTS "${GENERATED_DIR}/main_${problem_name}.c")
  add_executable(tracker_example "${GENERATED_DIR}/main_${problem_name}.c")
  target_link_libraries(tracker_example PRIVATE tracker_solver)
endif()
