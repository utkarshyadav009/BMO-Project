cmake_minimum_required(VERSION 3.20)

option(BMO_ENABLE_CUDA "Enable ggml CUDA backend when available" OFF)
option(BMO_TARGET_JETSON "Optimize build specifically for Jetson Orin Nano" OFF)

if(BMO_ENABLE_CUDA)
    project(bmo_minimal LANGUAGES C CXX CUDA)
else()
    project(bmo_minimal LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BMO_USE_FETCHCONTENT "Fetch llama.cpp if local submodule is missing" OFF)
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to llama.cpp source tree")
set(LLAMA_CPP_TAG "master" CACHE STRING "llama.cpp git tag/branch for FetchContent fallback")

if(BMO_ENABLE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "Enable ggml CUDA backend" FORCE)
    add_compile_definitions(BMO_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(BMO_TARGET_JETSON)
        set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA Architecture for Jetson Orin Nano" FORCE)
        add_compile_options(
            "$<$<COMPILE_LANGUAGE:C,CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:C,CXX>:-march=armv8.2-a>"
        )
    else()
        set(CMAKE_CUDA_ARCHITECTURES "90;87" CACHE STRING "CUDA Architectures" FORCE)
    endif()
endif()

if(EXISTS "${LLAMA_CPP_DIR}/ggml/CMakeLists.txt")
    message(STATUS "Using local llama.cpp at: ${LLAMA_CPP_DIR}")
    # Build only ggml subtree if present; this keeps build lighter than full llama app targets.
    add_subdirectory("${LLAMA_CPP_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml-build")
elseif(BMO_USE_FETCHCONTENT)
    include(FetchContent)
    message(STATUS "Local llama.cpp not found; fetching ${LLAMA_CPP_TAG}")
    FetchContent_Declare(
        llamacpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG        ${LLAMA_CPP_TAG}
    )
    FetchContent_MakeAvailable(llamacpp)

    if(EXISTS "${llamacpp_SOURCE_DIR}/ggml/CMakeLists.txt")
        add_subdirectory("${llamacpp_SOURCE_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml-build")
    else()
        message(FATAL_ERROR "Fetched llama.cpp does not contain ggml/CMakeLists.txt")
    endif()
else()
    message(FATAL_ERROR
        "Could not find llama.cpp at ${LLAMA_CPP_DIR}. "
        "Either add it as a submodule there or configure with -DBMO_USE_FETCHCONTENT=ON")
endif()

add_library(bmo STATIC bmo.cpp bmo_compute.cpp)

if(BMO_ENABLE_CUDA)
    target_sources(bmo PRIVATE bmo_cuda_kernels.cu)
    set_target_properties(bmo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Link bmo against CUDA runtime if CUDA is enabled
if(BMO_ENABLE_CUDA)
    target_link_libraries(bmo PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

# Main test binary (minimal correctness harness)
add_executable(bmo_minimal bmo_minimal.cpp)

# Main model loader binary
add_executable(bmo_main main.cpp)

# Header include paths (robust across local and fetched layouts).
if(EXISTS "${LLAMA_CPP_DIR}/ggml/include")
    target_include_directories(bmo PRIVATE "${LLAMA_CPP_DIR}/ggml/include")
    target_include_directories(bmo_minimal PRIVATE "${LLAMA_CPP_DIR}/ggml/include")
    target_include_directories(bmo_main PRIVATE "${LLAMA_CPP_DIR}/ggml/include")
    target_include_directories(bmo PRIVATE "${CMAKE_SOURCE_DIR}")
endif()
if(DEFINED llamacpp_SOURCE_DIR AND EXISTS "${llamacpp_SOURCE_DIR}/ggml/include")
    target_include_directories(bmo PRIVATE "${llamacpp_SOURCE_DIR}/ggml/include")
    target_include_directories(bmo_minimal PRIVATE "${llamacpp_SOURCE_DIR}/ggml/include")
    target_include_directories(bmo_main PRIVATE "${llamacpp_SOURCE_DIR}/ggml/include")
    target_include_directories(bmo PRIVATE "${CMAKE_SOURCE_DIR}")
endif()

# Also add project include dir so bmo.h is found
target_include_directories(bmo_minimal PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(bmo_main PRIVATE "${CMAKE_SOURCE_DIR}")

# Link any ggml targets available in this build.
set(_ggml_targets ggml ggml-base ggml-cpu ggml-cuda)
set(_link_targets "")
foreach(t IN LISTS _ggml_targets)
    if(TARGET ${t})
        list(APPEND _link_targets ${t})
    endif()
endforeach()

if(NOT _link_targets)
    message(FATAL_ERROR "No ggml targets were discovered. Check llama.cpp/ggml build output.")
endif()

# Link bmo library against ggml targets
target_link_libraries(bmo PRIVATE ${_link_targets})

message(STATUS "Linking bmo_minimal and bmo_main with targets: ${_link_targets}")
target_link_libraries(bmo_minimal PRIVATE ${_link_targets} bmo)
target_link_libraries(bmo_main PRIVATE ${_link_targets} bmo)

if(BMO_ENABLE_CUDA)
    # Standalone CUDA unpack kernel validation
    add_executable(bmo_unpack_kernel_test bmo_unpack_kernel_test.cu)
    target_include_directories(bmo_unpack_kernel_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_link_libraries(bmo_unpack_kernel_test PRIVATE ${_link_targets} bmo)
    set_target_properties(bmo_unpack_kernel_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
