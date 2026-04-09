#pragma once
#ifndef PCL_SEGEMENT_FEC_UNION_H
#define PCL_SEGEMENT_FEC_UNION_H

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <algorithm>

// 1. 结构体改名，防止与 FEC1.h 冲突
struct PointIndex_Tag_FEC_Union {
    int nPointIndex;
    int nNumberTag;
};

// 2. 比较函数改名，防止冲突
inline bool NumberTag_FEC_Union(const PointIndex_Tag_FEC_Union& p0, const PointIndex_Tag_FEC_Union& p1) {
    return p0.nNumberTag < p1.nNumberTag;
}

// 3. 极速版 FEC_Union (并查集重构)
std::vector<pcl::PointIndices> FEC_Union(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int min_component_size, double tolorance, int max_n) {
    using namespace std;

    if (cloud->size() < min_component_size) {
        return {};
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtreeflann;
    cloud_kdtreeflann.setInputCloud(cloud);

    const int cloud_size = static_cast<int>(cloud->size());

    // ==========================================
    // 并查集 (DSU) 初始化
    // ==========================================
    vector<int> parent(cloud_size);
    for (int i = 0; i < cloud_size; ++i) {
        parent[i] = i; 
    }

    auto find_root = [&](int i) {
        int root = i;
        while (root != parent[root]) root = parent[root];
        int curr = i;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    };

    auto unite = [&](int i, int j) {
        int root_i = find_root(i);
        int root_j = find_root(j);
        if (root_i != root_j) {
            parent[root_i] = root_j; 
        }
    };
    // ==========================================

    vector<int> pointIdx;
    vector<float> pointSquaredDistance;
    pointIdx.reserve(max_n);
    pointSquaredDistance.reserve(max_n);

    vector<bool> searched(cloud_size, false);

    for (int i = 0; i < cloud_size; ++i) {
        if (!searched[i]) {
            pointIdx.clear();
            pointSquaredDistance.clear();
            
            cloud_kdtreeflann.radiusSearch(cloud->points[i], tolorance, pointIdx, pointSquaredDistance, max_n);
            
            for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
                int idx = pointIdx[j];
                unite(i, idx);         
                searched[idx] = true;  
            }
        }
    }

    // ==========================================
    // 整理与提取
    // ==========================================
    vector<PointIndex_Tag_FEC_Union> indices_tags(cloud_size);
    for (int i = 0; i < cloud_size; ++i) {
        indices_tags[i].nPointIndex = i;
        indices_tags[i].nNumberTag = find_root(i); 
    }

    // 排序聚合
    sort(indices_tags.begin(), indices_tags.end(), NumberTag_FEC_Union);

    vector<pcl::PointIndices> cluster_indices;
    int begin_index = 0;
    for (int i = 0; i < cloud_size; ++i) {
        if (indices_tags[i].nNumberTag != indices_tags[begin_index].nNumberTag) {
            int cluster_size = i - begin_index;
            if (cluster_size >= min_component_size) {
                pcl::PointIndices inliers;
                inliers.indices.reserve(cluster_size);
                for (int j = begin_index; j < i; ++j) inliers.indices.push_back(indices_tags[j].nPointIndex);
                cluster_indices.push_back(inliers);
            }
            begin_index = i;
        }
    }

    // 处理最后一个收尾的聚类
    int final_cluster_size = cloud_size - begin_index;
    if (final_cluster_size >= min_component_size) {
        pcl::PointIndices inliers;
        inliers.indices.reserve(final_cluster_size);
        for (int j = begin_index; j < cloud_size; ++j) inliers.indices.push_back(indices_tags[j].nPointIndex);
        cluster_indices.push_back(inliers);
    }

    return cluster_indices;
}

#endif