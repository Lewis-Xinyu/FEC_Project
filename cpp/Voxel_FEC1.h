#pragma once
#ifndef PCL_SEGEMENT_VOXEL_FEC1_H
#define PCL_SEGEMENT_VOXEL_FEC1_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>

#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

struct VoxelKey_VFEC1 {
    int x, y, z;

    bool operator==(const VoxelKey_VFEC1& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash_VFEC1 {
    std::size_t operator()(const VoxelKey_VFEC1& k) const {
        return std::hash<int>{}(k.x)
             ^ (std::hash<int>{}(k.y) * 73856093u)
             ^ (std::hash<int>{}(k.z) * 19349663u);
    }
};

inline VoxelKey_VFEC1 pointToVoxel_VFEC1(const pcl::PointXYZ& pt, double voxel_size) {
    return VoxelKey_VFEC1{
        static_cast<int>(std::floor(pt.x / voxel_size)),
        static_cast<int>(std::floor(pt.y / voxel_size)),
        static_cast<int>(std::floor(pt.z / voxel_size))
    };
}

inline std::unordered_map<VoxelKey_VFEC1, std::vector<int>, VoxelHash_VFEC1>
buildVoxelGrid_VFEC1(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double voxel_size) {
    std::unordered_map<VoxelKey_VFEC1, std::vector<int>, VoxelHash_VFEC1> grid;
    for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
        VoxelKey_VFEC1 key = pointToVoxel_VFEC1(cloud->points[i], voxel_size);
        grid[key].push_back(i);
    }
    return grid;
}

inline void radiusSearch_VFEC1(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const std::unordered_map<VoxelKey_VFEC1, std::vector<int>, VoxelHash_VFEC1>& grid,
    int query_index,
    double tolerance,
    std::vector<int>& result_indices,
    int max_n = 0
) {
    result_indices.clear();
    const auto& p1 = cloud->points[query_index];
    const double tol_sqr = tolerance * tolerance;
    VoxelKey_VFEC1 base_k = pointToVoxel_VFEC1(p1, tolerance);

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                VoxelKey_VFEC1 nk{ base_k.x + dx, base_k.y + dy, base_k.z + dz };
                auto it = grid.find(nk);
                if (it == grid.end()) continue;

                for (int idx : it->second) {
                    const auto& p2 = cloud->points[idx];
                    double d_sqr =
                        (p1.x - p2.x) * (p1.x - p2.x) +
                        (p1.y - p2.y) * (p1.y - p2.y) +
                        (p1.z - p2.z) * (p1.z - p2.z);

                    if (d_sqr <= tol_sqr) {
                        result_indices.push_back(idx);
                        if (max_n > 0 && static_cast<int>(result_indices.size()) >= max_n) {
                            return;
                        }
                    }
                }
            }
        }
    }
}

struct PointIndex_Tag_VFEC1 {
    int nPointIndex;
    int nNumberTag;
};

inline bool NumberTag_VFEC1(const PointIndex_Tag_VFEC1& p0, const PointIndex_Tag_VFEC1& p1) {
    return p0.nNumberTag < p1.nNumberTag;
}

inline std::vector<pcl::PointIndices> Voxel_FEC1(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_component_size,
    double tolorance,
    int max_n
) {
    using Clock = std::chrono::high_resolution_clock;

    auto t_total_begin = Clock::now();
    double build_ms = 0.0, search_ms = 0.0, merge_ms = 0.0, final_ms = 0.0;

    if (!cloud || static_cast<int>(cloud->size()) < min_component_size) return {};

    auto t0 = Clock::now();
    auto grid = buildVoxelGrid_VFEC1(cloud, tolorance);
    auto t1 = Clock::now();
    build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int cloud_size = static_cast<int>(cloud->size());
    std::vector<int> marked_indices(cloud_size, 0);
    std::vector<int> pointIdx;

    int tag_num = 1;
    std::unordered_map<int, std::vector<int>> tag_to_indices;

    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] == 0) {
            auto ts0 = Clock::now();
            radiusSearch_VFEC1(cloud, grid, i, tolorance, pointIdx, max_n);
            auto ts1 = Clock::now();
            search_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

            int min_tag_num = tag_num;
            for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
                int idx = pointIdx[j];
                if (marked_indices[idx] > 0 && marked_indices[idx] < min_tag_num) {
                    min_tag_num = marked_indices[idx];
                }
            }

            auto tm0 = Clock::now();
            for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
                int idx = pointIdx[j];
                int current_tag = marked_indices[idx];

                if (current_tag > 0 && current_tag != min_tag_num) {
                    auto it = tag_to_indices.find(current_tag);
                    if (it != tag_to_indices.end()) {
                        auto& points = it->second;
                        for (int p : points) {
                            marked_indices[p] = min_tag_num;
                            tag_to_indices[min_tag_num].push_back(p);
                        }
                        tag_to_indices.erase(current_tag);
                    }
                }

                if (marked_indices[idx] != min_tag_num) {
                    marked_indices[idx] = min_tag_num;
                    tag_to_indices[min_tag_num].push_back(idx);
                }
            }
            auto tm1 = Clock::now();
            merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();

            if (min_tag_num == tag_num) ++tag_num;
        }
    }

    auto tf0 = Clock::now();

    std::vector<PointIndex_Tag_VFEC1> indices_tags(cloud_size);
    for (int i = 0; i < cloud_size; ++i) {
        indices_tags[i].nPointIndex = i;
        indices_tags[i].nNumberTag = marked_indices[i];
    }
    std::sort(indices_tags.begin(), indices_tags.end(), NumberTag_VFEC1);

    std::vector<pcl::PointIndices> cluster_indices;
    int begin_index = 0;
    for (int i = 0; i < cloud_size; ++i) {
        if (indices_tags[i].nNumberTag != indices_tags[begin_index].nNumberTag) {
            if ((i - begin_index) >= min_component_size) {
                pcl::PointIndices inliers;
                inliers.indices.reserve(i - begin_index);
                for (int j = begin_index; j < i; ++j) {
                    inliers.indices.push_back(indices_tags[j].nPointIndex);
                }
                cluster_indices.push_back(std::move(inliers));
            }
            begin_index = i;
        }
    }

    if ((cloud_size - begin_index) >= min_component_size) {
        pcl::PointIndices inliers;
        inliers.indices.reserve(cloud_size - begin_index);
        for (int j = begin_index; j < cloud_size; ++j) {
            inliers.indices.push_back(indices_tags[j].nPointIndex);
        }
        cluster_indices.push_back(std::move(inliers));
    }

    auto tf1 = Clock::now();
    final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

    auto t_total_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();

    std::cout << std::fixed << std::setprecision(3)
              << "FEC1_Voxel total= " << total_ms << " ms "
              << "build= " << build_ms << " ms "
              << "search= " << search_ms << " ms "
              << "merge= " << merge_ms << " ms "
              << "final= " << final_ms << " ms "
              << "clusters=" << cluster_indices.size() << "\n";

    return cluster_indices;
}

#endif