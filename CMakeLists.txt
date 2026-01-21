cmake_minimum_required(VERSION 3.14)
project(ChatAgent LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags for Ryzen AI Max+
# if(MSVC)
#    add_compile_options(/O2 /arch:AVX2)
# else()
#    add_compile_options(-O3 -march=native -funroll-loops)
# endif()

# --- Dependencies ---

# llama.cpp
set(LLAMA_BUILD_COMMON ON)
add_subdirectory(external/llama.cpp)

# HNSWLib
include_directories(external/hnswlib)

# Nlohmann JSON
add_subdirectory(external/json)

# --- Main Executable ---
add_executable(agent 
    src/main.cpp 
)

target_include_directories(agent PRIVATE src)

# Link Dependencies
target_link_libraries(agent PRIVATE 
    llama 
    nlohmann_json::nlohmann_json
)
