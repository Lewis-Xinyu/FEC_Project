#pragma once
#pragma warning(disable:4996)
#ifndef PCL_SEGEMENT_EC_H
#define PCL_SEGEMENT_EC_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

std::vector<pcl::PointIndices> EC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                 double cluster_tolerance, 
                                 int min_cluster_size) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_total_begin = Clock::now();

    auto tb0 = Clock::now();
    // 创建KdTree对象用于搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    auto tb1 = Clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

    // 执行欧几里得聚类
    auto ts0 = Clock::now();
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    auto ts1 = Clock::now();

    const double search_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();
    const double merge_ms = 0.0;
    const double final_ms = 0.0;
    const double total_ms = std::chrono::duration<double, std::milli>(ts1 - t_total_begin).count();

    std::cout << std::fixed << std::setprecision(3)
              << "EC total= " << total_ms << " ms "
              << "build= " << build_ms << " ms "
              << "search= " << search_ms << " ms "
              << "merge= " << merge_ms << " ms "
              << "final= " << final_ms << " ms "
              << "clusters=" << cluster_indices.size() << "\n";

    return cluster_indices;
}

#endif // PCL_SEGEMENT_EC_H
