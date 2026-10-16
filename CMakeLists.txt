cmake_minimum_required(VERSION 3.20)
project(numkit_signal CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NUMKIT_WITH_SIMD "Enable Highway SIMD acceleration" ON)

include(FetchContent)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NUMKIT_BUILD_TESTS ${_NUMKIT_TOP_BUILD_TESTS} CACHE BOOL "" FORCE)
endif()

if(NUMKIT_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    if(NOT TARGET numkit)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib")
            add_library(numkit STATIC IMPORTED)
            set_target_properties(numkit PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/Release/numkit.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/include"
                INTERFACE_COMPILE_DEFINITIONS "NUMKIT_CORE_BUILD=1"
                INTERFACE_LINK_LIBRARIES "numkit_value;numkit_sys;numkit_ops"
            )
        else()
            set(NUMKIT_BUILD_CLI OFF CACHE BOOL "" FORCE)
            set(NUMKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/CMakeLists.txt")
                add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../numkit" _deps/numkit)
            else()
                FetchContent_Declare(
                    numkit_core
                    GIT_REPOSITORY https://github.com/numkit/numkit.git
                    GIT_TAG main
                )
                FetchContent_MakeAvailable(numkit_core)
            endif()
            set(NUMKIT_BUILD_TESTS ON CACHE BOOL "" FORCE)
        endif()
    endif()

    if(NOT TARGET GTest::gtest_main)
        if(MSVC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib")
            add_library(GTest::gtest STATIC IMPORTED)
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest.lib"
                INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/googletest/include"
            )
            add_library(GTest::gtest_main STATIC IMPORTED)
            set_target_properties(GTest::gtest_main PROPERTIES
                IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/lib/Release/gtest_main.lib"
                INTERFACE_LINK_LIBRARIES "GTest::gtest"
            )
        else()
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
            set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src/CMakeLists.txt")
                set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${CMAKE_CURRENT_SOURCE_DIR}/../numkit/build/_deps/googletest-src" CACHE PATH "" FORCE)
            endif()
            FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG v1.14.0
            )
            FetchContent_MakeAvailable(googletest)
        endif()
    endif()

    file(GLOB SIGNAL_TEST_SOURCES "tests/*_test.cpp")
    add_executable(numkit_signal_test ${SIGNAL_TEST_SOURCES})
    target_include_directories(numkit_signal_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(numkit_signal_test PRIVATE
        numkit_signal_core
        numkit
        GTest::gtest_main
    )
    if(NUMKIT_WITH_SIMD)
        target_link_libraries(numkit_signal_test PRIVATE hwy hwy_contrib)
    endif()
    add_dependencies(numkit_signal_test numkit_signal)
    if(MSVC)
        target_compile_options(numkit_signal_test PRIVATE /W3 /MP /bigobj)
    endif()
    add_test(NAME numkit_signal_test COMMAND numkit_signal_test)
endif()
