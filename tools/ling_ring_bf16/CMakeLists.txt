cmake_minimum_required(VERSION 3.24)
project(strata_ring_reference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(ring_reference STATIC
  ${STRATA_ROOT}/src/ring/attention.cpp
  ${STRATA_ROOT}/src/ring/routing.cpp
  ${STRATA_ROOT}/src/ring/expert_cache.cpp
  ${STRATA_ROOT}/src/ring/tensor_store.cpp)
target_include_directories(ring_reference PUBLIC ${STRATA_ROOT}/include)
enable_testing()
foreach(unit attention routing expert_cache tensor_store numerics generation)
  add_executable(ring_${unit}_test ${STRATA_ROOT}/tests/ring_${unit}_test.cpp)
  target_link_libraries(ring_${unit}_test PRIVATE ring_reference)
  target_compile_options(ring_${unit}_test PRIVATE -UNDEBUG)
  add_test(NAME ring_${unit} COMMAND ring_${unit}_test)
endforeach()
option(RING_ENABLE_CUDA "Build opt-in BF16 Ring reference executable" OFF)
if(RING_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_executable(strata_ring_bf16
    ${STRATA_ROOT}/src/ring/engine.cu
    ${STRATA_ROOT}/src/kernels/cuda/native_bf16.cu)
  set_target_properties(strata_ring_bf16 PROPERTIES CUDA_ARCHITECTURES 89)
  target_compile_options(strata_ring_bf16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections>)
  target_link_options(strata_ring_bf16 PRIVATE -Wl,--gc-sections)
  target_link_libraries(strata_ring_bf16 PRIVATE ring_reference CUDA::cublas)
endif()
