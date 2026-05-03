cmake_minimum_required(VERSION 3.14)

project(FEC_Project LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    PCL REQUIRED COMPONENTS
    common
    io
    kdtree
    search
    segmentation
    filters
    features
    visualization
)
find_package(Eigen3 REQUIRED NO_MODULE)

if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()
endif()

find_package(OpenMP REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(fec_project_options INTERFACE)
target_include_directories(fec_project_options INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fec_project_options INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX ${PCL_LIBRARIES})

function(add_fec_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE fec_project_options)
endfunction()

add_fec_executable(fec_run main.cpp)
add_fec_executable(fec_legacy_demo_run main_old.cpp)
add_fec_executable(fec_union_tolerance_run main_new.cpp)
add_fec_executable(kitti_eval_run main_kitti_eval.cpp)
add_fec_executable(semanticposs_eval_run main_semanticposs_eval.cpp)
add_fec_executable(semanticposs_noground_eval_run main_semanticposs_noground_eval.cpp)
add_fec_executable(local_report_eval_run main_local_report_eval.cpp)
add_fec_executable(semanticposs_report_eval_run main_semanticposs_report_eval.cpp)
add_fec_executable(semanticposs_docx_eval_run main_semanticposs_docx_eval.cpp)
add_fec_executable(semanticposs_tune_eval_run main_semanticposs_tune_eval.cpp)
add_fec_executable(semanticposs_fusion_build_run main_semanticposs_fusion_build.cpp)
add_fec_executable(semanticposs_fusion_time_run main_semanticposs_fusion_time.cpp)
