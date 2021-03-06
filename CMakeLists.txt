cmake_minimum_required(VERSION 3.0)

# Variables that can/must be passed to the cmake command:
#
# - MPI is required. It is best to pass MPI wrappers as C and CXX compilers to CMake.
# - -DNOOMP=1 to compile without OpenMP (default build is with OpenMP)
# - -DNOVECTORIZE=1 to disable vectorization in GCC and Clang (takes precedence over other options).
# - -DSSE=1 to compile with SSE 4.2 instructions (default)
# - -DAVX=1 to compile with AVX instructions.
# - -DAVX_2=1 to compile with AVX 2.0.
# - -DZEN=1 to compiler for AMD Zen with AVX 2.0.
# - -DSKYLAKE=1 to compile with AVX-512 instructions for Xeon Skylake CPUs.
# - -DKNL=1 to compile for Xeon Phi Knights Landing
# - -DMICKNC=1 to compile for Xeon Phi Knights Corner (deprecated).
# - -DPROFILE=1 for profiling with gprof.
# - -DSLURM=1 for running the automated tests on a system managed by Slurm
# - -DSLURMTESTTHREADS=<n> for configuring multi-threaded tests to use n threads in case of Slurm
#
# - EIGEN3_ROOT needs to be set to the path of the Eigen top-level directory.
# - -DBUILD_BLOCK_SIZE=<n> for some integer n, to build the block solver operations for
#     an additional block size of <n>. By default, block sizes 4 and 5 are supported.

project (struct3d)

# detect C++ compiler - the C compiler must be the same type too
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CXX_COMPILER_GNUCLANG TRUE)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	set(CXX_COMPILER_GNU TRUE)
	message(STATUS "GNU C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "GNU"))
	  message(SEND_ERROR "C and C++ compilers should match!")
	endif()
  else()
	set(CXX_COMPILER_CLANG TRUE)
	message(STATUS "Clang C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "Clang"))
	  message(SEND_ERROR "C and C++ compilers should match!")
	endif()
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  set(CXX_COMPILER_INTEL TRUE)
  message(STATUS "Intel C++ compiler detected")
  if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "Intel"))
	message(SEND_ERROR "C and C++ compilers should match!")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
  set(CXX_COMPILER_CRAY TRUE)
  message(STATUS "Cray C++ compiler detected")
  if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "Cray"))
	message(SEND_ERROR "C and C++ compilers should match!")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "PGI")
  set(CXX_COMPILER_PGI TRUE)
  message(STATUS "PGI C++ compiler detected")
  if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "PGI"))
	message(SEND_ERROR "C and C++ compilers should match!")
  endif()
else()
  message(SEND_ERROR "Could not identify compiler!")
endif()

# Basic options
if(CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
  set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Werror")
  set (CMAKE_C_FLAGS "-std=c11 -Wall -Werror")
  set (CMAKE_CXX_FLAGS_DEBUG "-g")
  set (CMAKE_CXX_FLAGS_RELEASE "-O3")
  if( CXX_COMPILER_CLANG )
    # assume libcxx has been built with Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif()
elseif(CXX_COMPILER_PGI)
  set (CMAKE_CXX_FLAGS "-std=c++14")
  set (CMAKE_C_FLAGS "-std=c11")
  set (CMAKE_CXX_FLAGS_DEBUG "-g")
  set (CMAKE_CXX_FLAGS_RELEASE "-O2")
elseif(CXX_COMPILER_CRAY)
  set (CMAKE_CXX_FLAGS "-h std=c++14 -h error_on_warning")
  set (CMAKE_C_FLAGS "-h std=c11 -h error_on_warning")
  set (CMAKE_CXX_FLAGS_DEBUG "-g -O1")
  set (CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

if(NOVECTORIZE)
  # As release/debug flags are added after shared flags, we need to add to debug and release flags.
  if(CXX_COMPILER_CLANG)
	set (CMAKE_CXX_FLAGS_RELEASE
	  "${CMAKE_CXX_FLAGS_RELEASE} -fno-tree-vectorize -fno-slp-vectorize -fno-vectorize")
	set (CMAKE_CXX_FLAGS_DEBUG
	  "${CMAKE_CXX_FLAGS_DEBUG} -fno-tree-vectorize -fno-slp-vectorize -fno-vectorize")
    message(WARNING "Vectorization disabled.")
  elseif(CXX_COMPILER_GNU)
	set (CMAKE_CXX_FLAGS_RELEASE
	  "${CMAKE_CXX_FLAGS_RELEASE} -fno-tree-vectorize -fno-tree-slp-vectorize")
	set (CMAKE_CXX_FLAGS_DEBUG
	  "${CMAKE_CXX_FLAGS_DEBUG} -fno-tree-vectorize -fno-tree-slp-vectorize")
    message(WARNING "Vectorization disabled.")
  elseif(CXX_COMPILER_INTEL)
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -no-vec")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -no-vec")
    message(WARNING "Vectorization disabled.")
  else()
	message(WARNING "Don't know flags to disable vectorization for this compiler!")
  endif()
endif(NOVECTORIZE)

if(NOFORCESIMD)
  # Disable OpenMP SIMD in SGS application kernels
  add_definitions(-DNOFORCESIMD)
  message(STATUS "OpenMP SIMD will not be used for SGS application")
endif()

# to compile with OpenMP
if(NOOMP)
  if(CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
  endif()
  if(CXX_COMPILER_PGI)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nomp")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nomp")
  endif()
  if(CXX_COMPILER_CRAY)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hnoomp")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -hnoomp")
  endif()
  message(STATUS "Compiling without OpenMP")
else()
  if(CXX_COMPILER_GNUCLANG)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
	message(STATUS "Compiling with OpenMP for GNU")
  elseif(CXX_COMPILER_INTEL)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")	  
	message(STATUS "Compiling with OpenMP for Intel")
  elseif(CXX_COMPILER_CRAY)
	message(STATUS "Compiling with OpenMP for Cray")
  elseif(CXX_COMPILER_PGI)
	message(STATUS "Compiling with OpenMP for PGI")
  else()
	message(SEND_ERROR "Don't know OpenMP flag for this compiler!")
  endif()
endif()

# set verbosity options
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-inline-vec-optimized-omp=optimizations.info")
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=inline -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize")
elseif(CXX_COMPILER_INTEL)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=3")
elseif(CXX_COMPILER_CRAY)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hlist=a -hreport=fisv")
endif()

# Architecture to target
if(MICKNC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
  message(STATUS "Compiling for KNC")
  if(NOT CXX_COMPILER_INTEL)
	message(SEND_ERROR "Need to use Intel compiler for Xeon Phi!")
  endif()
  
else()
  # Select hardware-specific optimizations
  if(AVX)
	if(CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
	elseif(CXX_COMPILER_PGI)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=sandybridge")
	elseif(CXX_COMPILER_CRAY)
	endif()
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	  message(STATUS "Compiling with AVX")
	endif()
  elseif(AVX_2)
	if(CXX_COMPILER_INTEL)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx2")
	elseif(CXX_COMPILER_GNUCLANG)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
	elseif(CXX_COMPILER_PGI)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=haswell")
	elseif(CXX_COMPILER_CRAY)
	  message(STATUS "Architecture options for Cray are to be set by the compiler wrapper.")
	  add_definitions(-DEIGEN_VECTORIZE_AVX2 -DEIGEN_VECTORIZE_FMA)
	else()
	  message(SEND_ERROR "Don't know/require option for this compiler!")
	endif()
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	  message(STATUS "Compiling with AVX-2")
	endif()
  elseif(SKYLAKE)
	if(CXX_COMPILER_INTEL)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx512 -qopt-zmm-usage=high")
	elseif(CXX_COMPILER_GNUCLANG)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=skylake-avx512")
	elseif(CXX_COMPILER_PGI)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=skylake")
	elseif(CXX_COMPILER_CRAY)
	  message(STATUS "Architecture options for Cray are to be set by the compiler wrapper.")
	else()
	  message(SEND_ERROR "Don't know option for this compiler!")
	endif()
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	  message(STATUS "Compiling with AVX-512 for Skylake")
	endif()
  elseif(KNL)
	add_definitions(-DEIGEN_VECTORIZE_AVX512 -DEIGEN_VECTORIZE_FMA)
	if(CXX_COMPILER_INTEL)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xmic-avx512")
	elseif(CXX_COMPILER_GNUCLANG)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=knl")
	elseif(CXX_COMPILER_PGI)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=knl")
	elseif(CXX_COMPILER_CRAY)
	  message(STATUS "Architecture options for Cray are to be set by the compiler wrapper.")
	else()
	  message(SEND_ERROR "Don't know/require option for this compiler!")
	endif()
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	  message(STATUS "Compiling with AVX-512 for KNL")
	endif()
  elseif(ZEN)
	if(CXX_COMPILER_INTEL)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx2")
	elseif(CXX_COMPILER_GNUCLANG)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=znver1")
	elseif(CXX_COMPILER_PGI)
	  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=zen")
	elseif(CXX_COMPILER_CRAY)
	  message(STATUS "Architecture options for Cray are to be set by the compiler wrapper.")
	else()
	  message(SEND_ERROR "Don't know option for this compiler!")
	endif()
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
	  message(STATUS "Compiling with AVX 2.0 for AMD Zen")
	endif()
  else()
	message(STATUS "No specific vectorization requested, using default")
  endif()

endif()

# ---------------- Libraries -------------------------------------------- #

# Boost for aligned allocation etc.
find_package(Boost 1.60 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

# PETSc
# If PETSc variables were not passed to CMake, set them from environment variables
if(NOT DEFINED PETSC_DIR)
  set(PETSC_DIR $ENV{PETSC_DIR} CACHE PATH "PETSc install directory")
endif()
if(NOT DEFINED PETSC_ARCH)
  set(PETSC_ARCH $ENV{PETSC_ARCH} CACHE STRING "PETSc architecture to use")
endif()
find_library(PETSC_LIB NAMES petsc PATHS ${PETSC_DIR}/${PETSC_ARCH}/lib
  DOC "Location of PETSc library")
include_directories(${PETSC_DIR}/include ${PETSC_DIR}/${PETSC_ARCH}/include)
message(STATUS "Building with PETSc found at ${PETSC_LIB}")

# BLASTed
if(WITH_BLASTED)
  find_path(BLASTED_INCLUDE_DIR blasted_petsc.h PATHS ENV BLASTED_DIR PATH_SUFFIXES include
	DOC "Include directory of BLASTed library")
  if(${BLASTED_INCLUDE_DIR} STREQUAL "BLASTED_INCLUDE_DIR-NOTFOUND")
	message(SEND_ERROR "BLASTed requested, but header not found!")
  endif()
  include_directories(${BLASTED_INCLUDE_DIR})

  find_library(BLASTED_LIB blasted_petsc PATHS ENV BLASTED_BIN_DIR PATH_SUFFIXES lib
	DOC "Location of BLASTed library")
  if(${BLASTED_LIB} STREQUAL "BLASTED_LIB-NOTFOUND")
	message(SEND_ERROR "BLASTed requested, but library not found!")
  endif()

  message(STATUS "Building with BLASTed: ${BLASTED_INCLUDE_DIR}, ${BLASTED_LIB}")
  add_definitions(-DHAVE_BLASTED)

  # Harwell's matrix ordering library - static library is assumed
  if(WITH_MC64)
	if(NOT DEFINED MC64_DIR)
	  set(MC64_DIR $ENV{MC64_DIR} CACHE PATH "MC64 install directory")
	endif()
	find_library(MC64_LIB libmc64.a ${MC64_DIR}/lib)
	add_definitions(-DHAVE_MC64)
	message(STATUS "Building with MC64 found at ${MC64_LIB}")
	if(CXX_COMPILER_GNUCLANG)
	  set(FORTLDFLAGS "-lgfortran")
	elseif(CXX_COMPILER_INTEL)
	  set(FORTLDFLAGS "-lifcore")
	else()
	  message(WARNING "Don't know Fortran library for this compiler")
	endif()
  endif(WITH_MC64)
endif(WITH_BLASTED)

# -------------------------------------------------------------------------- #

if(SLURM)
  set(SEQEXEC "srun")
  set(SEQTASKS "-n 1")
  if(SLURMTESTTHREADS)
	  set(THREADOPTS "--cpus-per-task=${SLURMTESTTHREADS}")
  else()
	  set(THREADOPTS "--cpus-per-task=4")
  endif()
else()
  set(SEQEXEC "")
  set(SEQTASKS "")
  set(THREADOPTS "")
endif(SLURM)

if(SLURM)
  set(MPIEXEC "srun")
  set(MPIOPTS "--cpu-bind=cores")
else()
  set(MPIEXEC "mpirun")
  set(MPIOPTS "")
endif(SLURM)

# ---------------------------------------------------------------------------- #

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UDEBUG -DNDEBUG")

# profiling
if(PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  message(STATUS "Debug mode")
else()
  message(STATUS "Release mode")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src)

# Tests
enable_testing()
add_subdirectory(test)
