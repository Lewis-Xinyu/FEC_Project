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

std::vector<pcl::PointIndices> RG(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                 int min_cluster_size,
                                 int number_of_neighbors,
                                 float smoothness_threshold,
                                 float curvature_threshold) {
    // 估计法线
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    // 区域生长分割
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

    return clusters;
}

#endif // PCL_SEGEMENT_RG_H