cmake_minimum_required(VERSION 3.19.1)
if(APPLE)
#https://stackoverflow.com/questions/29902862/how-to-get-cmake-to-use-the-default-compiler-on-system-path
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} clang PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} clang++ PATHS ENV PATH NO_DEFAULT_PATH)
endif()

project(sips)

# if not specified by user, the standard build type is release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release Coverage." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CXX)
SET(CMAKE_CXX_FLAGS "-march=native -mavx -O3 -g -Wall -pedantic -std=c++14")
#ADD_DEFINITIONS(-O3)
#ADD_DEFINITIONS(-std=c++17)
#ADD_DEFINITIONS(-Wall)
#ADD_DEFINITIONS(-g)

#cmake_policy(SET CMP0015 NEW)

#set OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add and compile the gtest library
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# set the hyperalg include directory
set(hyperalg_include ../../include)
include_directories(${hyperalg_include})
message("hyperalg include directory ${hyperalg_include}")

## build the hyperalg library
#FILE(GLOB hyperalg_sources ${hyperalg_include}/*.c*)
#add_library(hyperalg_lib ${hyperalg_sources})
## add hyperalg_lib to target_link_libraries

# get all the source files
FILE(GLOB sources *.c*)
message("test files ${sources}")

add_executable(test_main ${sources})

target_link_libraries(test_main GTest::gtest_main pthread)


include(GoogleTest)
gtest_discover_tests(test_main)
