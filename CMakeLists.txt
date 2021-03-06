CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(N3LDGStanceDetector)

SET(CMAKE_CXX_COMPILER g++)

SET(SRC_DIR src)

function(find_mkl)
  set(MKL_ARCH intel64)
  set(MKL_ROOT /opt/intel/mkl)
  find_path(MKL_INCLUDE_DIR mkl.h
            PATHS ${MKL_ROOT} ${MKL_ROOT}/include)
  find_library(MKL_CORE_LIB NAMES mkl_intel_lp64 mkl_sequential mkl_core
               PATHS ${MKL_ROOT} ${MKL_ROOT}/lib/${MKL_ARCH}
               DOC "MKL core library path")

  find_library(MKL_COMPILER_LIB NAMES iomp5 libiomp5md
               PATHS ${MKL_ROOT} ${MKL_ROOT}/../compiler/lib/${MKL_ARCH}              #Windows
                     ${MKL_ROOT}/../compilers_and_libraries/linux/lib/${MKL_ARCH}_lin #Linux
               DOC "MKL compiler lib (for threaded MKL)")

  if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_COMPILER_LIB)
    get_filename_component(MKL_CORE_LIB_DIR ${MKL_CORE_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_DIR ${MKL_COMPILER_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_FILE ${MKL_COMPILER_LIB} NAME)
    message(STATUS "Found MKL\n   * include: ${MKL_INCLUDE_DIR},\n   * core library dir: ${MKL_CORE_LIB_DIR},\n   * compiler library: ${MKL_COMPILER_LIB}")

    # Due to a conflict with /MT and /MD, MSVC needs mkl_intel_lp64 linked last, or we can change individual
    # projects to use /MT (mkl_intel_lp64 linked with /MT, default MSVC projects use /MD), or we can instead
    # link to the DLL versions. For now I'm opting for this solution which seems to work with projects still
    # at their default /MD. Linux build requires the mkl_intel_lp64 to be linked first. So...:
    if(MSVC)
      set(LIBS ${LIBS} mkl_sequential mkl_core mkl_intel_lp64 ${MKL_COMPILER_LIB_FILE} PARENT_SCOPE)
    else()
      set(LIBS ${LIBS} mkl_intel_lp64 mkl_sequential mkl_core ${MKL_COMPILER_LIB_FILE} PARENT_SCOPE)
    endif()
    include_directories(${MKL_INCLUDE_DIR})
    link_directories(${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR})
    set(MKL_LINK_DIRS ${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR} PARENT_SCOPE) # Keeping this for python build
  else()
    message(FATAL_ERROR "Failed to find MKL in path: ${MKL_ROOT} (Did you set MKL_ROOT properly?)")
  endif()
endfunction()

######## Cross-compiler, cross-platform options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH")
if (MKL OR MKL_ROOT)
  find_mkl()  # sets include/lib directories and sets ${LIBS} needed for linking
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
endif()

IF(MEMORY MATCHES cnmem)
    ADD_DEFINITIONS(-DDEVICE_MEMORY=0)
ELSEIF(MEMORY MATCHES cuda)
    ADD_DEFINITIONS(-DDEVICE_MEMORY=1)
ELSE()
    ADD_DEFINITIONS(-DDEVICE_MEMORY=2)
ENDIF()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    ADD_DEFINITIONS(-DN3LDG_DEBUG)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w -msse3 -funroll-loops  -std=c++11 -O1" )
else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -w -std=c++11 -O2 -march=native" )
endif()
IF(MEMORY MATCHES cnmem)
    ADD_DEFINITIONS(-DDEVICE_MEMORY=0)
ELSEIF(MEMORY MATCHES cuda)
    ADD_DEFINITIONS(-DDEVICE_MEMORY=1)
ELSE()
    ADD_DEFINITIONS(-DDEVICE_MEMORY=2)
ENDIF()

IF(TEST_CUDA)
    ADD_DEFINITIONS(-DTEST_CUDA)
ENDIF()
INCLUDE_DIRECTORIES(N3LDG/N3LDG/tensor N3LDG/eigen N3LDG/N3LDG ${SRC_DIR}/basic ${SRC_DIR}/classifier/lstm)
IF(USE_CUDA)
    ADD_DEFINITIONS(-DUSE_GPU)
    INCLUDE_DIRECTORIES(N3LDG/cnmem/include N3LDG/cuda N3LDG/include)
    SET(LIBS ${LIBS} n3ldg_cuda)
    INCLUDE_DIRECTORIES(/usr/local/cuda/samples/common/inc)
    FIND_PACKAGE(CUDA REQUIRED)
    FIND_CUDA_HELPER_LIBS(curand)
    SET(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
    MESSAGE("CUDA_INCLUDE: " ${CUDA_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})
    SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
    MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    SET(LIBS ${LIBS} ${CUDA_LIBRARIES})
    ADD_SUBDIRECTORY(N3LDG/cnmem)
    ADD_SUBDIRECTORY(N3LDG/cuda)
    SET(LIBS ${LIBS} n3ldg_cuda)
ENDIF()

AUX_SOURCE_DIRECTORY(${SRC_DIR} SRCS)
AUX_SOURCE_DIRECTORY(${SRC_DIR}/basic SRCS)
AUX_SOURCE_DIRECTORY(${SRC_DIR}/classifier SRCS)
ADD_EXECUTABLE(classifier ${SRCS})
TARGET_LINK_LIBRARIES(classifier ${LIBS})
