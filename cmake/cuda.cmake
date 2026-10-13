# Copyright (c) OpenMMLab. All rights reserved.

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    # suppress 'CMAKE_CUDA_ARCHITECTURES' warning
    cmake_policy(SET CMP0104 OLD)
endif ()

if (MSVC OR (NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY))
    # use shared, on windows, python api can't build with static lib.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
endif ()

if (MSVC)
    # no plugin in BuildCustomizations and no specify cuda toolset
    if (NOT CMAKE_VS_PLATFORM_TOOLSET_CUDA)
        message(FATAL_ERROR "Please install CUDA MSBuildExtensions")
    endif ()

    if (CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR)
        # find_package(CUDA) required ENV{CUDA_PATH}
        set(ENV{CUDA_PATH} ${CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR})
    else ()
        # we use CUDA_PATH and ignore nvcc.exe
        # cmake will import highest cuda props version, which may not equal to CUDA_PATH
        if (NOT (DEFINED ENV{CUDA_PATH}))
            message(FATAL_ERROR "Please set CUDA_PATH environment variable")
        endif ()

        string(REGEX REPLACE ".*v([0-9]+)\\..*" "\\1" _MAJOR $ENV{CUDA_PATH})
        string(REGEX REPLACE ".*v[0-9]+\\.([0-9]+).*" "\\1" _MINOR $ENV{CUDA_PATH})
        if (NOT (${CMAKE_VS_PLATFORM_TOOLSET_CUDA} STREQUAL "${_MAJOR}.${_MINOR}"))
            message(FATAL_ERROR "Auto detected cuda version ${CMAKE_VS_PLATFORM_TOOLSET_CUDA}"
                " is mismatch with ENV{CUDA_PATH} $ENV{CUDA_PATH}. Please modify CUDA_PATH"
                " to match ${CMAKE_VS_PLATFORM_TOOLSET_CUDA} or specify cuda toolset by"
                " cmake -T cuda=/path/to/cuda ..")
        endif ()

        if (NOT (DEFINED ENV{CUDA_PATH_V${_MAJOR}_${_MINOR}}))
            message(FATAL_ERROR "Please set CUDA_PATH_V${_MAJOR}_${_MINOR} environment variable")
        endif ()
    endif ()
endif ()

# nvcc compiler settings
find_package(CUDA REQUIRED)

if (MSVC)
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819,/wd4828")
    if (HAVE_CXX_FLAG_UTF_8)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8")
    endif ()
else ()
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
    # Explicitly set the cuda host compiler. Because the default host compiler #
    # selected by cmake maybe wrong.
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_FLAGS
            "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC,-Wall,-fvisibility=hidden")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-gnu-unique")
    endif ()
endif ()

enable_language(CUDA)

# set virtual compute architecture and real ones
if (CUDA_VERSION_MAJOR EQUAL 11)
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_37,code=sm_37")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_53,code=sm_53")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    if (CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    endif ()
    if (CUDA_VERSION VERSION_GREATER_EQUAL "11.5")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_87,code=sm_87")
    endif ()
    if (CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90")
    endif ()
elseif (CUDA_VERSION_MAJOR EQUAL 12)
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_53,code=sm_53")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
    # set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_87,code=sm_87")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")
    set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90")
    if (CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_100,code=sm_100")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_101,code=sm_101")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_120,code=sm_120")
    endif ()
    if (CUDA_VERSION VERSION_GREATER_EQUAL "12.9")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_103,code=sm_103")
        set(_NVCC_FLAGS "${_NVCC_FLAGS} -gencode arch=compute_121,code=sm_121")
    endif ()
endif ()

set(CMAKE_CUDA_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMMDEPLOY_USE_CUDA=1")

if (NOT MSVC)
    set(CMAKE_CUDA_STANDARD 14)
endif ()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_NVCC_FLAGS}")

if (MSVC AND MMDEPLOY_USE_MSCV_STATIC)
    string(REPLACE -MD -MT CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}")
    string(REPLACE -MD -MT CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
endif ()
