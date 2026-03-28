#ShaderCmakeFile
cmake_minimum_required(VERSION 3.11)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(BMO_Face_Engine)

include(FetchContent)

# -----------------------------------------------------------------------------
# 1. PLATFORM CONFIGURATION
# -----------------------------------------------------------------------------
if(EMSCRIPTEN)
    set(PLATFORM "Web" CACHE STRING "Raylib Platform" FORCE)
    set(GRAPHICS "GRAPHICS_API_OPENGL_ES3" CACHE STRING "Raylib Graphics" FORCE)
else()
    set(PLATFORM "Desktop" CACHE STRING "Raylib Platform" FORCE)
    set(GRAPHICS "GRAPHICS_API_OPENGL_ES2" CACHE STRING "Raylib Graphics" FORCE)
endif()

# 2. Fetch Dependencies
FetchContent_Declare(raylib URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz)
FetchContent_MakeAvailable(raylib)



FetchContent_Declare(raygui URL https://github.com/raysan5/raygui/archive/refs/tags/4.0.tar.gz)
FetchContent_MakeAvailable(raygui)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

# -----------------------------------------------------------------------------
# 3. SOURCE FILES
# -----------------------------------------------------------------------------
add_executable(BMO_Engine
    src/SFacePoserFinal.cpp
    src/eyes_es.fs
    src/brow_es.fs
    src/tears_es.fs
    src/mouth_es.fs
    src/pixelizer_es.fs
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(BMO_Engine PRIVATE -Wno-enum-compare)
endif()

# -----------------------------------------------------------------------------
# 4. LINK LIBRARIES (Separated by Platform!)
# -----------------------------------------------------------------------------
if(EMSCRIPTEN)
    # WEB BUILD: No X11 or Pthreads allowed here
    target_link_libraries(BMO_Engine PRIVATE raylib nlohmann_json::nlohmann_json)
else()
    # DESKTOP BUILD: Needs Linux system libraries
    target_link_libraries(BMO_Engine PRIVATE raylib nlohmann_json::nlohmann_json m pthread dl rt X11)
endif()

# 5. Include Directories
target_include_directories(BMO_Engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${raygui_SOURCE_DIR}/src
)

# -----------------------------------------------------------------------------
# 6. DESKTOP POST-BUILD ASSETS (Ignored for Web)
# -----------------------------------------------------------------------------
if(NOT EMSCRIPTEN)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})

    add_custom_command(TARGET BMO_Engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/mouth_es.fs" "$<TARGET_FILE_DIR:BMO_Engine>/mouth_es.fs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/eyes_es.fs" "$<TARGET_FILE_DIR:BMO_Engine>/eyes_es.fs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/brow_es.fs" "$<TARGET_FILE_DIR:BMO_Engine>/brow_es.fs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/tears_es.fs" "$<TARGET_FILE_DIR:BMO_Engine>/tears_es.fs"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/pixelizer_es.fs" "$<TARGET_FILE_DIR:BMO_Engine>/pixelizer_es.fs"
    )
endif()

# -----------------------------------------------------------------------------
# 7. WEB BUILD SETTINGS (EMSCRIPTEN)
# -----------------------------------------------------------------------------
if(EMSCRIPTEN)
    set(ASSET_STAGING "/tmp/bmo_assets")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${ASSET_STAGING})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/eyes_es.fs
              ${CMAKE_CURRENT_SOURCE_DIR}/src/brow_es.fs
              ${CMAKE_CURRENT_SOURCE_DIR}/src/tears_es.fs
              ${CMAKE_CURRENT_SOURCE_DIR}/src/mouth_es.fs
              ${CMAKE_CURRENT_SOURCE_DIR}/src/pixelizer_es.fs
         DESTINATION ${ASSET_STAGING})
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/face_database.txt DESTINATION ${ASSET_STAGING})
    
    target_compile_definitions(raylib PRIVATE SUPPORT_GAMEPAD=0)
    target_compile_definitions(BMO_Engine PRIVATE SUPPORT_GAMEPAD=0)
         
    target_link_options(BMO_Engine PRIVATE
        # -sUSE_GLFW=3
        -sWASM=1
        -sALLOW_MEMORY_GROWTH=1
        -sMIN_WEBGL_VERSION=2
        -sMAX_WEBGL_VERSION=2
        -sASYNCIFY
        -sASYNCIFY_STACK_SIZE=65536
        -sENVIRONMENT=web
        -sASSERTIONS=1
        -sDISABLE_EXCEPTION_CATCHING=0
        "SHELL:--shell-file /tmp/bmo_shell.html"
        "--pre-js" "${CMAKE_CURRENT_SOURCE_DIR}/gamepad_fix.js"
        "SHELL:--preload-file ${ASSET_STAGING}/assets@assets"
        "SHELL:--preload-file ${ASSET_STAGING}/eyes_es.fs@eyes_es.fs"
        "SHELL:--preload-file ${ASSET_STAGING}/brow_es.fs@brow_es.fs"
        "SHELL:--preload-file ${ASSET_STAGING}/tears_es.fs@tears_es.fs"
        "SHELL:--preload-file ${ASSET_STAGING}/mouth_es.fs@mouth_es.fs"
        "SHELL:--preload-file ${ASSET_STAGING}/pixelizer_es.fs@pixelizer_es.fs"
        "SHELL:--preload-file /tmp/bmo_assets/face_database.txt@face_database.txt"
       # "SHELL:--preload-file ../face_database.txt@face_database.txt"
     )
    set_target_properties(BMO_Engine PROPERTIES SUFFIX ".html")
endif()
#https://1801016-2.tumblr.com/post/183847530563/lip-sync-mouth-chart-examples-here-are-some-mouth



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
