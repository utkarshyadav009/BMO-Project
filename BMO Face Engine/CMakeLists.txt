#ShaderCmakeFile
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

# 2. Fetch Raygui
FetchContent_Declare(
    raygui
    URL https://github.com/raysan5/raygui/archive/refs/tags/4.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(raygui)

# -----------------------------------------------------------------------------
# SOURCE FILES
# -----------------------------------------------------------------------------
# Since FacePoser.cpp includes ParametricMouth.cpp directly, we only list FacePoser here.
add_executable(BMO_Engine src/SFacePoser.cpp)

# Silence the "enum-compare" warnings from raygui
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(BMO_Engine PRIVATE -Wno-enum-compare)
endif()

# 3. Link Libraries
target_link_libraries(BMO_Engine
  PRIVATE
    raylib
    winmm
    # cppzmq  <-- Uncomment if you installed ZeroMQ
    # libzmq
)

# 4. Include Directories
# [UPDATED] Added 'src' so headers/files inside src can be found easily
target_include_directories(BMO_Engine PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${raygui_SOURCE_DIR}/src 
)

# -----------------------------------------------------------------------------
# ASSETS & SHADERS
# -----------------------------------------------------------------------------
# Copy the assets folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})

# [NEW] Copy the shader from src to the build folder
# This ensures LoadShader(0, "mouth_es.fs") works when you run the app.
# This automatically puts the file next to the .exe (in /Debug/ or /Release/)
add_custom_command(TARGET BMO_Engine POST_BUILD
    # Copy Mouth Shader
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mouth_es.fs"
    "$<TARGET_FILE_DIR:BMO_Engine>/mouth_es.fs"

    # Copy Eye Shader [NEW]
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/src/eyes_es.fs"
    "$<TARGET_FILE_DIR:BMO_Engine>/eyes_es.fs"
)







# cmake_minimum_required(VERSION 3.11)
# set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# project(BMO_Face_Engine)

# set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# include(FetchContent)

# # 1. Fetch Raylib
# FetchContent_Declare(
#     raylib
#     URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz
#     DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# )
# FetchContent_MakeAvailable(raylib)

# # 2. Fetch Raygui (NEW STEP)
# FetchContent_Declare(
#     raygui
#     URL https://github.com/raysan5/raygui/archive/refs/tags/4.0.tar.gz
#     DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# )
# FetchContent_MakeAvailable(raygui)

# add_executable(BMO_Engine src/FacePoser.cpp)
# # [NEW] Silence the "enum-compare" warnings from raygui
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(BMO_Engine PRIVATE -Wno-enum-compare)
# endif()

# # 3. Link Libraries
# # (Assuming you actually need ZeroMQ, otherwise remove those lines if not installed)
# # find_package(cppzmq CONFIG REQUIRED) 
# # find_package(ZeroMQ CONFIG REQUIRED)

# target_link_libraries(BMO_Engine
#   PRIVATE
#     raylib
#     winmm
#     # cppzmq  <-- Only uncomment if you actually installed ZeroMQ
#     # libzmq
# )

# # 4. Include Directories
# # You must tell CMake where to find the downloaded raygui.h
# target_include_directories(BMO_Engine PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${raygui_SOURCE_DIR}/src 
# )

# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})