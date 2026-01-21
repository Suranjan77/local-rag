cmake_minimum_required(VERSION 4.0)

project(ChatAgent LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force use llvm's lib instead of GCC's libstd++ (which gets chosen by default in Linux)
# However, this requires installing libc++ in the OS
add_compile_options(-stdlib=libc++)
add_link_options(-stdlib=libc++)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# llama.cpp
set(LLAMA_BUILD_COMMON ON)
add_subdirectory(external/llama.cpp)

# Nlohmann JSON
add_subdirectory(external/json)

add_executable(agent
        src/main.cpp
)

target_include_directories(agent PRIVATE
        src
        external/hnswlib
)

target_link_libraries(agent PRIVATE
        llama
        nlohmann_json::nlohmann_json
)