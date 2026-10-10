cmake_minimum_required(VERSION 3.20)
project(nk_signal CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NK_WITH_SIMD "Enable Highway SIMD acceleration" OFF)

# Dependency: numkit/value
include(FetchContent)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
    message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/nk_value)
else()
    FetchContent_Declare(
        nk_value
        GIT_REPOSITORY https://github.com/numkit/value.git
        GIT_TAG main
    )
    FetchContent_MakeAvailable(nk_value)
endif()

file(GLOB_RECURSE SIGNAL_SOURCES
    "src/convolution/*.cpp"
    "src/digital_filtering/*.cpp"
    "src/filter_analysis/*.cpp"
    "src/filter_design/*.cpp"
    "src/filter_implementation/*.cpp"
    "src/measurements/*.cpp"
    "src/multirate/*.cpp"
    "src/smoothing/*.cpp"
    "src/spectral_analysis/*.cpp"
    "src/time_frequency/*.cpp"
    "src/transforms/*.cpp"
    "src/waveform_generation/*.cpp"
    "src/windows/*.cpp"
    "src/register/*.cpp"
    "src/shims/*.cpp"
    "src/module.cpp"
)

if(NOT NK_WITH_SIMD)
    list(FILTER SIGNAL_SOURCES EXCLUDE REGEX ".*_highway\\.cpp$")
else()
    FetchContent_Declare(
        highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG 1.2.0
    )
    FetchContent_MakeAvailable(highway)
    list(FILTER SIGNAL_SOURCES EXCLUDE REGEX ".*_portable\\.cpp$")
endif()

add_library(nk_signal SHARED ${SIGNAL_SOURCES})

target_compile_definitions(nk_signal PRIVATE NK_PLUGIN_BUILD=1)

if(NK_WITH_SIMD)
    target_compile_definitions(nk_signal PUBLIC NK_WITH_SIMD=1)
    target_link_libraries(nk_signal PRIVATE hwy hwy_contrib)
endif()

target_include_directories(nk_signal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(nk_signal PUBLIC nk_value)

if(MSVC)
    target_compile_options(nk_signal PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(nk_signal PRIVATE -Wall -Wextra -fPIC)
endif()
