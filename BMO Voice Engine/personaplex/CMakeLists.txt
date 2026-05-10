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
        add_compile_definitions(BMO_TARGET_JETSON)
        add_compile_definitions(BMO_JETSON)
        # Default to sm_87 for actual Jetson Orin builds, but allow overriding from
        # the CLI (e.g. `-DCMAKE_CUDA_ARCHITECTURES=90` on a non-Jetson box used to
        # debug the Jetson code path). Drop FORCE so user values win.
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA Architecture for Jetson Orin Nano")
        endif()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            add_compile_options(
                "$<$<COMPILE_LANGUAGE:C,CXX>:-O3>"
                "$<$<COMPILE_LANGUAGE:C,CXX>:-march=armv8.2-a>"
            )
        endif()
    else()
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "90;87" CACHE STRING "CUDA Architectures")
        endif()
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

    # Fused dequant + matmul prototype (telemetry vs unpack+baseline path)
    add_executable(bmo_fused_mat_test bmo_fused_mat_test.cu)
    target_include_directories(bmo_fused_mat_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_link_libraries(bmo_fused_mat_test PRIVATE ${_link_targets} bmo)
    set_target_properties(bmo_fused_mat_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# -----------------------------------------------------------------------------
# Shared library bridge (Phase 4.1) -- libbmo.so for Python / external clients.
# Builds the same source set as the static `bmo` lib plus the C-ABI shim
# (bmo_api.cpp). bmo_cuda_kernels.cu is only added when CUDA is enabled.
# -----------------------------------------------------------------------------

find_package(Threads REQUIRED)

set(_bmo_shared_sources bmo.cpp bmo_compute.cpp bmo_api.cpp)
if(BMO_ENABLE_CUDA)
    list(APPEND _bmo_shared_sources bmo_cuda_kernels.cu)
endif()

add_library(bmo_shared SHARED ${_bmo_shared_sources})
set_target_properties(bmo_shared PROPERTIES
    OUTPUT_NAME              bmo
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET    hidden
    VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(bmo_shared PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if(EXISTS "${LLAMA_CPP_DIR}/ggml/include")
    target_include_directories(bmo_shared PRIVATE "${LLAMA_CPP_DIR}/ggml/include")
endif()
if(DEFINED llamacpp_SOURCE_DIR AND EXISTS "${llamacpp_SOURCE_DIR}/ggml/include")
    target_include_directories(bmo_shared PRIVATE "${llamacpp_SOURCE_DIR}/ggml/include")
endif()

# Tell bmo_api.h to use dllexport on Windows when building the shared lib.
target_compile_definitions(bmo_shared PRIVATE BMO_BUILDING_SHARED=1)

if(BMO_TARGET_JETSON)
    target_compile_definitions(bmo_shared PRIVATE BMO_JETSON=1 BMO_TARGET_JETSON=1)
endif()
if(BMO_ENABLE_CUDA)
    target_compile_definitions(bmo_shared PRIVATE BMO_ENABLE_CUDA=1)
    set_target_properties(bmo_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(BMO_TARGET_JETSON)
        set_property(TARGET bmo_shared PROPERTY CUDA_ARCHITECTURES 87)
    endif()
    target_link_libraries(bmo_shared PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

target_link_libraries(bmo_shared PRIVATE ${_link_targets} Threads::Threads)

message(STATUS "bmo_shared sources: ${_bmo_shared_sources}")
message(STATUS "bmo_shared linking: ${_link_targets} Threads::Threads")
