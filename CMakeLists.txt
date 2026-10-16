cmake_minimum_required(VERSION 3.20)
project(numkit_signal CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)

include(FetchContent)

# Dependency: numkit/value
if(NOT TARGET numkit_value)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../value/CMakeLists.txt")
        message(STATUS "Using local numkit/value from ${CMAKE_CURRENT_SOURCE_DIR}/../value")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../value" _deps/numkit_value)
    else()
        FetchContent_Declare(
            numkit_value
            GIT_REPOSITORY https://github.com/numkit/value.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_value)
    endif()
endif()

# Dependency: numkit/sys
if(NOT TARGET numkit_sys)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../sys/CMakeLists.txt")
        message(STATUS "Using local numkit/sys from ${CMAKE_CURRENT_SOURCE_DIR}/../sys")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../sys" _deps/numkit_sys)
    else()
        FetchContent_Declare(
            numkit_sys
            GIT_REPOSITORY https://github.com/numkit/sys.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_sys)
    endif()
endif()

# Dependency: numkit/ops
if(NOT TARGET numkit_ops)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ops/CMakeLists.txt")
        message(STATUS "Using local numkit/ops from ${CMAKE_CURRENT_SOURCE_DIR}/../ops")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ops" _deps/numkit_ops)
    else()
        FetchContent_Declare(
            numkit_ops
            GIT_REPOSITORY https://github.com/numkit/ops.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(numkit_ops)
    endif()
endif()

file(GLOB_RECURSE SIGNAL_COMPUTE_SOURCES
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
)

file(GLOB_RECURSE SIGNAL_PLUGIN_SOURCES
    "src/register/*.cpp"
    "src/module.cpp"
)

if(NOT NUMKIT_WITH_SIMD)
    list(FILTER SIGNAL_COMPUTE_SOURCES EXCLUDE REGEX ".*_highway\\.cpp$")
else()
    list(FILTER SIGNAL_COMPUTE_SOURCES EXCLUDE REGEX ".*_portable\\.cpp$")
endif()

add_library(numkit_signal_core STATIC ${SIGNAL_COMPUTE_SOURCES})
add_library(numkit::signal_core ALIAS numkit_signal_core)
set_target_properties(numkit_signal_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(numkit_signal_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_signal_core PUBLIC numkit_value numkit_sys numkit_ops)

if(NUMKIT_WITH_SIMD)
    target_compile_definitions(numkit_signal_core PUBLIC NUMKIT_WITH_SIMD=1)
    target_link_libraries(numkit_signal_core PRIVATE hwy hwy_contrib)
endif()

if(MSVC)
    target_compile_options(numkit_signal_core PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_signal_core PRIVATE -Wall -Wextra -fPIC)
endif()

# Shared plugin module
add_library(numkit_signal SHARED ${SIGNAL_PLUGIN_SOURCES})
add_library(numkit::signal ALIAS numkit_signal)

target_compile_definitions(numkit_signal PRIVATE NUMKIT_PLUGIN_BUILD=1)

target_include_directories(numkit_signal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(numkit_signal PUBLIC numkit_signal_core)

if(MSVC)
    target_compile_options(numkit_signal PRIVATE /W4 /MP /bigobj)
else()
    target_compile_options(numkit_signal PRIVATE -Wall -Wextra -fPIC)
endif()
