cmake_minimum_required(VERSION 3.11)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(BMO_Face_Engine)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
include(FetchContent)

# 1. Fetch Raylib
FetchContent_Declare(
    raylib
    URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(raylib)

# 2. Fetch Raygui (NEW STEP)
FetchContent_Declare(
    raygui
    URL https://github.com/raysan5/raygui/archive/refs/tags/4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(raygui)

add_executable(BMO_Engine src/FacePoser.cpp)

# 3. Link Libraries
# (Assuming you actually need ZeroMQ, otherwise remove those lines if not installed)
# find_package(cppzmq CONFIG REQUIRED) 
# find_package(ZeroMQ CONFIG REQUIRED)

target_link_libraries(BMO_Engine
  PRIVATE
    raylib
    winmm
    # cppzmq  <-- Only uncomment if you actually installed ZeroMQ
    # libzmq
)

# 4. Include Directories
# You must tell CMake where to find the downloaded raygui.h
target_include_directories(BMO_Engine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${raygui_SOURCE_DIR}/src 
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})