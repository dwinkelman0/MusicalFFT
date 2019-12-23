# Place all code in ./src
# Build from ./build using cmake ..

# Environment configurations
cmake_minimum_required (VERSION 3.10)
set (CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 11)
add_definitions(-DBOOST_NO_CXX11_SCOPED_ENUMS)

# Project configurations
project (MusicalFFT)
set (MusicalFFT_VERSION_MAJOR 0)
set (MusicalFFT_VERSION_MINOR 0)
set (TargetName_MusicalFFT musicalfft)
set (TargetName_Tests ${TargetName_MusicalFFT}-tests)

# Include project headers
include_directories (include)

# Build a linked library
file(GLOB MUSICALFFT_SOURCES "src/*.cpp")
add_library(${TargetName_MusicalFFT} SHARED ${MUSICALFFT_SOURCES})

# Link to Boost
# https://stackoverflow.com/questions/3897839/how-to-link-c-program-with-boost-using-cmake
find_package(Boost 1.65 COMPONENTS program_options filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
target_link_libraries(${TargetName_MusicalFFT} LINK_PUBLIC ${Boost_LIBRARIES} libOpenCL.so)

# Unit tests
include(GoogleTest)
file(GLOB TESTS_SOURCES "tests/*.cpp")
add_executable(${TargetName_Tests} ${TESTS_SOURCES})
target_link_libraries(${TargetName_Tests} ${TargetName_MusicalFFT} gtest pthread)