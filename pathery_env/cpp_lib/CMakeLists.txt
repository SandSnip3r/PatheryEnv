cmake_minimum_required(VERSION 3.10)

# Project name
project(PathfindingProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Target for the shared library
add_library(pathfinding STATIC pathfinder.cpp pathfinding.cpp)

# Executable target
add_executable(main_exec main.cpp)

# Link the library to the executable
target_link_libraries(main_exec pathfinding)

# Set output directories for shared library and executable
set_target_properties(pathfinding PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(main_exec PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
