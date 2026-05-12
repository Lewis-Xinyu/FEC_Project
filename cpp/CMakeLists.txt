cmake_minimum_required(VERSION 3.14)
project(FEC_Project)
set(CMAKE_BUILD_TYPE "Release")

# 1. 寻找核心点云库 PCL (你已经成功了)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 2. Mac 专属外挂：强行指路 Homebrew 的 OpenMP
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)

# 3. 编译并链接
add_executable(fec_run main.cpp)
target_link_libraries(fec_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(kitti_eval_run main_kitti_eval.cpp)
target_link_libraries(kitti_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_eval_run main_semanticposs_eval.cpp)
target_link_libraries(semanticposs_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_noground_eval_run main_semanticposs_noground_eval.cpp)
target_link_libraries(semanticposs_noground_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(local_report_eval_run main_local_report_eval.cpp)
target_link_libraries(local_report_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_report_eval_run main_semanticposs_report_eval.cpp)
target_link_libraries(semanticposs_report_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_docx_eval_run main_semanticposs_docx_eval.cpp)
target_link_libraries(semanticposs_docx_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_tune_eval_run main_semanticposs_tune_eval.cpp)
target_link_libraries(semanticposs_tune_eval_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_fusion_time_run main_semanticposs_fusion_time.cpp)
target_link_libraries(semanticposs_fusion_time_run ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(semanticposs_cluster_viz_run
    main_semanticposs_cluster_viz.cpp
)

target_link_libraries(semanticposs_cluster_viz_run
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)