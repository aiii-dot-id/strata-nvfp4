cmake_minimum_required(VERSION 3.24)
project(strata_ling_reference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_library(ling_reference STATIC
  ${STRATA_ROOT}/src/ling/kda.cpp
  ${STRATA_ROOT}/src/ling/mla.cpp
  ${STRATA_ROOT}/src/ring/routing.cpp
  ${STRATA_ROOT}/src/ring/expert_cache.cpp
  ${STRATA_ROOT}/src/ring/tensor_store.cpp)
target_include_directories(ling_reference PUBLIC ${STRATA_ROOT}/include)
enable_testing()
foreach(unit kda mla)
  add_executable(ling_${unit}_test ${STRATA_ROOT}/tests/ling_${unit}_test.cpp)
  target_link_libraries(ling_${unit}_test PRIVATE ling_reference)
  target_compile_options(ling_${unit}_test PRIVATE -UNDEBUG)
  add_test(NAME ling_${unit} COMMAND ling_${unit}_test)
endforeach()
option(LING_ENABLE_CUDA "Build opt-in BF16 Ling reference executable" OFF)
if(LING_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_executable(strata_ling_bf16
    ${STRATA_ROOT}/src/ling/engine.cu
    ${STRATA_ROOT}/src/kernels/cuda/native_bf16.cu)
  set_target_properties(strata_ling_bf16 PROPERTIES CUDA_ARCHITECTURES 89)
  target_compile_options(strata_ling_bf16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections>)
  target_link_options(strata_ling_bf16 PRIVATE -Wl,--gc-sections)
  target_link_libraries(strata_ling_bf16 PRIVATE ling_reference)
endif()
