cmake_minimum_required(VERSION 3.11)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(BMO_Face_Engine)

# Use Raylib 5.0 (Critical for modern CMake compatibility)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
include(FetchContent)

FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(raylib)

add_executable(BMO_Engine src/main.cpp)
find_package(cppzmq CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)

target_link_libraries(BMO_Engine
  PRIVATE
    raylib
    winmm
    cppzmq
    libzmq
)
target_include_directories(BMO_Engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})