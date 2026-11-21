cmake_minimum_required(VERSION 3.26)
project(fastopt VERSION 0.1.0 LANGUAGES CXX)

# ==============================================================================
# 1. C++ Standard and Base Compiler Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

include(FetchContent)

# ==============================================================================
# 2. Offline Header-Only Eigen3 Setup
# ==============================================================================
FetchContent_Declare(
    Eigen3
    URL "${CMAKE_CURRENT_SOURCE_DIR}/externals/eigen-3.4.0.zip"
    URL_HASH SHA256=EBA3F3D414D2F8CBA2919C78EC6DAAB08FC71BA2BA4AE502B7E5D4D99FC02CDA
    SOURCE_SUBDIR "cmake_dummy_skip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Eigen3)

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3_Eigen INTERFACE)
    target_include_directories(Eigen3_Eigen SYSTEM INTERFACE "${eigen3_SOURCE_DIR}")
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
endif()

# ==============================================================================
# 3. Hardware Acceleration (SIMD: AVX2 + FMA) and OpenMP Multi-Threading
# ==============================================================================
option(FASTOPT_ENABLE_SIMD "Enable AVX2 and FMA vectorization instructions" ON)
option(FASTOPT_ENABLE_BLAS "Enable optional Intel MKL or OpenBLAS acceleration for Eigen GEMM" ON)
option(FASTOPT_BLAS_STATIC "Prefer static OpenBLAS linking (libopenblas.a) on MinGW for a self-contained .pyd" ON)

add_library(fastopt_compiler_flags INTERFACE)

if(MSVC)
    find_package(OpenMP REQUIRED)
    target_compile_options(fastopt_compiler_flags INTERFACE /O2 /fp:fast /EHsc /utf-8)
    if(FASTOPT_ENABLE_SIMD)
        target_compile_options(fastopt_compiler_flags INTERFACE /arch:AVX2)
    endif()
    target_link_libraries(fastopt_compiler_flags INTERFACE OpenMP::OpenMP_CXX)

elseif(MINGW)
    # Static linking of libstdc++, libgcc, libgomp, and libwinpthread in the exact
    # archive dependency order so _fastopt_impl.pyd has zero dependency on libwinpthread-1.dll.
    target_compile_options(fastopt_compiler_flags INTERFACE -O3 -ffast-math -fno-finite-math-only -Wall -Wextra -fopenmp)
    if(FASTOPT_ENABLE_SIMD)
        target_compile_options(fastopt_compiler_flags INTERFACE -march=native -mavx2 -mfma)
    endif()
    target_link_options(fastopt_compiler_flags INTERFACE -static-libgcc -static-libstdc++ -Wl,--exclude-libs,ALL)
    target_link_libraries(fastopt_compiler_flags INTERFACE
        "-Wl,-Bstatic" stdc++ gomp winpthread gcc_eh gcc "-Wl,-Bdynamic"
    )

else()
    # Linux / macOS
    find_package(OpenMP REQUIRED)
    target_compile_options(fastopt_compiler_flags INTERFACE -O3 -ffast-math -fno-finite-math-only -Wall -Wextra)
    if(FASTOPT_ENABLE_SIMD)
        target_compile_options(fastopt_compiler_flags INTERFACE -march=native -mavx2 -mfma)
    endif()
    target_link_libraries(fastopt_compiler_flags INTERFACE OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
endif()

# ==============================================================================
# 4. Python & CasADi (IPOPT + MUMPS Runtime) Discovery
# ==============================================================================
find_package(Python 3.9 COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, os; s = importlib.util.find_spec('casadi'); print(os.path.dirname(s.origin) if s and s.origin else '')"
    OUTPUT_VARIABLE FASTOPT_CASADI_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(FASTOPT_CASADI_DIR AND EXISTS "${FASTOPT_CASADI_DIR}")
    file(TO_CMAKE_PATH "${FASTOPT_CASADI_DIR}" FASTOPT_CASADI_DIR_CMAKE)
    message(STATUS "Found CasADi package directory for IPOPT/MUMPS: ${FASTOPT_CASADI_DIR_CMAKE}")
else()
    set(FASTOPT_CASADI_DIR_CMAKE "")
    message(STATUS "CasADi not found at CMake configure time; IPOPT/MUMPS will be discovered from 'casadi' at runtime.")
endif()

# ==============================================================================
# 5. Build C++ Core Static Library (fastopt_core)
# ==============================================================================
set(FASTOPT_CORE_SOURCES
    src/graph.cpp
    src/ops.cpp
    src/interp.cpp
    src/evaluator.cpp
    src/ipopt_solver.cpp
)

set(FASTOPT_BINDING_SOURCES
    src/bindings.cpp
)

add_library(fastopt_core STATIC ${FASTOPT_CORE_SOURCES})
target_include_directories(fastopt_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
if(FASTOPT_CASADI_DIR_CMAKE)
    target_compile_definitions(fastopt_core PUBLIC FASTOPT_DEFAULT_CASADI_DIR="${FASTOPT_CASADI_DIR_CMAKE}")
    if(EXISTS "${FASTOPT_CASADI_DIR_CMAKE}/include")
        target_include_directories(fastopt_core SYSTEM PUBLIC "${FASTOPT_CASADI_DIR_CMAKE}/include")
        target_compile_definitions(fastopt_core PUBLIC FASTOPT_HAS_IPOPT_HEADER=1)
    endif()
endif()
# Optional external BLAS (Intel MKL / OpenBLAS) for large neural network GEMM acceleration.
# Gracefully falls back to header-only Eigen AVX2+FMA GEMM if no BLAS library is installed.
if(FASTOPT_ENABLE_BLAS)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "fastopt: Enabling Intel MKL acceleration (EIGEN_USE_MKL_ALL)")
        target_compile_definitions(fastopt_core PUBLIC EIGEN_USE_MKL_ALL)
        target_link_libraries(fastopt_core PUBLIC MKL::MKL)
    else()
        set(FASTOPT_BLAS_CONFIGURED OFF)
        if(MINGW AND FASTOPT_BLAS_STATIC)
            # Derive the active MSYS2/MinGW sysroot lib directory from CMAKE_CXX_COMPILER
            # so VS Code / PowerShell CMake runs automatically locate ucrt64/lib/libopenblas.a.
            get_filename_component(_FASTOPT_CXX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
            get_filename_component(_FASTOPT_MINGW_ROOT "${_FASTOPT_CXX_BIN_DIR}" DIRECTORY)
            set(_FASTOPT_MINGW_LIB_HINTS
                "${_FASTOPT_MINGW_ROOT}/lib"
                "$ENV{MINGW_PREFIX}/lib"
                "C:/msys64/ucrt64/lib"
                "D:/msys64/ucrt64/lib"
            )
            find_library(FASTOPT_OPENBLAS_STATIC_LIB
                NAMES libopenblas.a
                HINTS ${_FASTOPT_MINGW_LIB_HINTS}
            )
            if(FASTOPT_OPENBLAS_STATIC_LIB)
                message(STATUS "fastopt: Enabling STATIC OpenBLAS acceleration (EIGEN_USE_BLAS): ${FASTOPT_OPENBLAS_STATIC_LIB}")
                target_compile_definitions(fastopt_core PUBLIC EIGEN_USE_BLAS)
                target_link_libraries(fastopt_core PUBLIC "${FASTOPT_OPENBLAS_STATIC_LIB}")
                find_library(FASTOPT_GFORTRAN_STATIC_LIB
                    NAMES libgfortran.a
                    HINTS ${_FASTOPT_MINGW_LIB_HINTS}
                )
                find_library(FASTOPT_QUADMATH_STATIC_LIB
                    NAMES libquadmath.a
                    HINTS ${_FASTOPT_MINGW_LIB_HINTS}
                )
                if(FASTOPT_GFORTRAN_STATIC_LIB)
                    target_link_libraries(fastopt_core PUBLIC "${FASTOPT_GFORTRAN_STATIC_LIB}")
                endif()
                if(FASTOPT_QUADMATH_STATIC_LIB)
                    target_link_libraries(fastopt_core PUBLIC "${FASTOPT_QUADMATH_STATIC_LIB}")
                endif()
                set(FASTOPT_BLAS_CONFIGURED ON)
            endif()
        endif()
        if(NOT FASTOPT_BLAS_CONFIGURED)
            find_package(BLAS QUIET)
            if(BLAS_FOUND)
                message(STATUS "fastopt: Enabling external BLAS acceleration (EIGEN_USE_BLAS): ${BLAS_LIBRARIES}")
                target_compile_definitions(fastopt_core PUBLIC EIGEN_USE_BLAS)
                target_link_libraries(fastopt_core PUBLIC ${BLAS_LIBRARIES})
            else()
                message(STATUS "fastopt: No external BLAS/MKL found; using built-in Eigen AVX2+FMA GEMM.")
            endif()
        endif()
    endif()
endif()

target_link_libraries(fastopt_core PUBLIC
    Eigen3::Eigen
    fastopt_compiler_flags
)

# ==============================================================================
# 6. Build nanobind Python Extension Module (_fastopt_impl)
# ==============================================================================
FetchContent_Declare(
    nanobind
    URL "${CMAKE_CURRENT_SOURCE_DIR}/externals/nanobind.zip"
    URL_HASH SHA256=AF87F5EBB5386CA6D09B9A525AF18FC080BBF9C61BF5B0480B3EB17A9C4B8A8E
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(
    _fastopt_impl
    STABLE_ABI
    NB_STATIC
    ${FASTOPT_BINDING_SOURCES}
)

target_link_libraries(_fastopt_impl PRIVATE fastopt_core)

# Output compiled extension directly into python/fastopt package directory
set_target_properties(_fastopt_impl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/python/fastopt>"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/python/fastopt>"
)

install(TARGETS _fastopt_impl
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/python/fastopt"
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/python/fastopt"
)
