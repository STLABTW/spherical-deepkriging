cmake_minimum_required(VERSION 3.10)
project(spherical_basis)

# === C++ Config ===
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Python & Pybind11 ===
# Use Python3_EXECUTABLE if provided (e.g., from venv), otherwise auto-detect
if(DEFINED Python3_EXECUTABLE)
    message(STATUS "Using provided Python executable: ${Python3_EXECUTABLE}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/../../../../venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/../../../../venv/bin/python")
    message(STATUS "Auto-detected Python from venv: ${Python3_EXECUTABLE}")
else()
    message(STATUS "Using system Python (auto-detected)")
endif()
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# === Spectra ===
# Spectra is header-only and depends on Eigen
# Use FetchContent to download if not available
include(FetchContent)
FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG v1.0.1
)

FetchContent_MakeAvailable(spectra)
set(SPECTRA_INCLUDE_DIR ${spectra_SOURCE_DIR}/include)
message(STATUS "Spectra include directory: ${SPECTRA_INCLUDE_DIR}")

# === Eigen (header-only) ===
# Use FetchContent to avoid requiring system Eigen headers at build time.
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)
set(EIGEN3_INCLUDE_DIRS ${eigen_SOURCE_DIR})

# === Module Build ===
pybind11_add_module(spherical_basis spherical_basis.cpp)

# === Compiler/Linker Flags ===
# Spectra (header-only, just need includes)
if(SPECTRA_INCLUDE_DIR)
    target_include_directories(spherical_basis PRIVATE ${SPECTRA_INCLUDE_DIR})
endif()

# Eigen include dir (set above)
if(EIGEN3_INCLUDE_DIRS)
    target_include_directories(spherical_basis PRIVATE ${EIGEN3_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "EIGEN3_INCLUDE_DIRS not set. This is a CMake configuration error.")
endif()

# === Output Location ===
set_target_properties(spherical_basis PROPERTIES
    OUTPUT_NAME "spherical_basis"
)
