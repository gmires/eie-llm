cmake_minimum_required(VERSION 3.16)
project(eie-llm VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default a Debug per ora (più facile fare debug)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_executable(eie-llm
  src/main.cpp
  src/gguf.cpp
  src/tokenizer.cpp
  src/ops.cpp
  src/ops_avx2.cpp
  src/cpuinfo.cpp
  src/model.cpp
  src/shell.cpp
  src/server.cpp
  src/bench.cpp
  src/prefix_cache.cpp
)

# Compila ops_avx2.cpp con -mavx2 -mfma per abilitare le intrinseche AVX2.
# Le altre translation unit rimangono generiche (compatibili con CPU senza AVX2).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/ops_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# Ora gli header stanno in include/ separati dai sorgenti
target_include_directories(eie-llm PRIVATE 
  include
  third_party
)

find_package(Threads REQUIRED)

# OpenMP — parallelizzazione multicore per matvec e ops vettoriali
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_compile_options(eie-llm PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(eie-llm PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(eie-llm PRIVATE
  -Wall -Wextra
  $<$<CONFIG:Debug>:-g -O0>
  $<$<CONFIG:Release>:-O3>
)

target_link_libraries(eie-llm PRIVATE Threads::Threads)