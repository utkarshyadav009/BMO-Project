cmake_minimum_required(VERSION 3.11)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(BMO_Speech_Engine)

include(FetchContent)

if(EMSCRIPTEN)
    set(PLATFORM "Web" CACHE STRING "Raylib Platform" FORCE)
    set(GRAPHICS "GRAPHICS_API_OPENGL_ES3" CACHE STRING "Raylib Graphics" FORCE)
else()
    set(PLATFORM "Desktop" CACHE STRING "Raylib Platform" FORCE)
    set(GRAPHICS "GRAPHICS_API_OPENGL_ES2" CACHE STRING "Raylib Graphics" FORCE)
endif()

set(FACE_ENGINE_SRC "/home/bmo/BMO-Project/BMO Face Engine/src")
set(FACE_ENGINE_ASSETS "/home/bmo/BMO-Project/BMO Face Engine/assets")
set(FACE_ENGINE_ROOT "/home/bmo/BMO-Project/BMO Face Engine")

FetchContent_Declare(raylib URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz)
FetchContent_MakeAvailable(raylib)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(BMO_Speech
    ${FACE_ENGINE_SRC}/FaceEngineV2.cpp
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(BMO_Speech PRIVATE -Wno-enum-compare)
endif()

if(EMSCRIPTEN)
    target_link_libraries(BMO_Speech PRIVATE raylib nlohmann_json::nlohmann_json)
else()
    target_link_libraries(BMO_Speech PRIVATE raylib nlohmann_json::nlohmann_json m pthread dl rt X11)
endif()

target_include_directories(BMO_Speech PRIVATE
    ${FACE_ENGINE_SRC}
    ${CMAKE_BINARY_DIR}/_deps/json-src/single_include/nlohmann
)

if(EMSCRIPTEN)
    set(ASSET_STAGING "/tmp/bmo_speech_assets")
    file(COPY ${FACE_ENGINE_ASSETS} DESTINATION ${ASSET_STAGING})
    file(COPY ${FACE_ENGINE_SRC}/mouth_es.fs DESTINATION ${ASSET_STAGING})
    file(COPY ${FACE_ENGINE_ROOT}/visemes_database.txt DESTINATION ${ASSET_STAGING})

    target_compile_definitions(raylib PRIVATE SUPPORT_GAMEPAD=0)
    target_compile_definitions(BMO_Speech PRIVATE SUPPORT_GAMEPAD=0)

    target_link_options(BMO_Speech PRIVATE
        -sWASM=1
        -sALLOW_MEMORY_GROWTH=1
        -sMIN_WEBGL_VERSION=2
        -sMAX_WEBGL_VERSION=2
        -sASYNCIFY
        -sASYNCIFY_STACK_SIZE=65536
        -sENVIRONMENT=web
        -sASSERTIONS=1
        -sEXPORTED_RUNTIME_METHODS=['HEAPF32','ccall','cwrap']
        "SHELL:--shell-file /tmp/bmo_speech_shell.html"
        "SHELL:--pre-js /tmp/gamepad_fix.js"
        "SHELL:--preload-file ${ASSET_STAGING}/assets@assets"
        "SHELL:--preload-file ${ASSET_STAGING}/mouth_es.fs@mouth_es.fs"
        "SHELL:--preload-file ${ASSET_STAGING}/visemes_database.txt@visemes_database.txt"
    )
    set_target_properties(BMO_Speech PROPERTIES SUFFIX ".html")
endif()
