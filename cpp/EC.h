#pragma once
#pragma warning(disable:4996)
#ifndef PCL_SEGEMENT_EC_H
#define PCL_SEGEMENT_EC_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <vector>

std::vector<pcl::PointIndices> EC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                 double cluster_tolerance, 
                                 int min_cluster_size) {
    // 创建KdTree对象用于搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // 执行欧几里得聚类
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    return cluster_indices;
}

#endif // PCL_SEGEMENT_EC_H
