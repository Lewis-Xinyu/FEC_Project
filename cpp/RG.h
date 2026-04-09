#pragma once
#pragma warning(disable:4996)
#ifndef PCL_SEGEMENT_RG_H
#define PCL_SEGEMENT_RG_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

std::vector<pcl::PointIndices> RG(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                 int min_cluster_size,
                                 int number_of_neighbors,
                                 float smoothness_threshold,
                                 float curvature_threshold) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_total_begin = Clock::now();

    auto tb0 = Clock::now();
    // 估计法线
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    auto tb1 = Clock::now();

    auto ts0 = Clock::now();
    normal_estimator.compute(*normals);
    auto ts1 = Clock::now();

    // 区域生长分割
    auto tm0 = Clock::now();
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(min_cluster_size);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(number_of_neighbors);
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(smoothness_threshold / 180.0 * M_PI);
    reg.setCurvatureThreshold(curvature_threshold);

    // 提取聚类
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);
    auto tm1 = Clock::now();

    const double build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
    const double search_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    const double merge_ms = std::chrono::duration<double, std::milli>(tm1 - tm0).count();
    const double final_ms = 0.0;
    const double total_ms = std::chrono::duration<double, std::milli>(tm1 - t_total_begin).count();

    std::cout << std::fixed << std::setprecision(3)
              << "RG total= " << total_ms << " ms "
              << "build= " << build_ms << " ms "
              << "search= " << search_ms << " ms "
              << "merge= " << merge_ms << " ms "
              << "final= " << final_ms << " ms "
              << "clusters=" << clusters.size() << "\n";

    return clusters;
}

#endif // PCL_SEGEMENT_RG_H
