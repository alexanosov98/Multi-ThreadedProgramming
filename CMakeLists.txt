cmake_minimum_required(VERSION 3.10)
project(Multi-ThreadedProgramming)

set(CMAKE_CXX_STANDARD 14)

# Add your source files
add_executable(SampleClient
        MapReduceFramework.h
        MapReduceClient.cpp
        MapReduceClient.h
        MapReduceClient.cpp
        ./Barrier/Barrier.cpp
        ./Barrier/Barrier.h
        MapReduceFramework.cpp
        SampleClient.cpp)


# Link the pthread library
target_link_libraries(SampleClient pthread)