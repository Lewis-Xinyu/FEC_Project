#pragma once
#ifndef PCL_SEGEMENT_FEC_TIMED_H
#define PCL_SEGEMENT_FEC_TIMED_H

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/PointIndices.h>

#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstddef>

struct FECStageStats {
    double total_ms  = 0.0;
    double build_ms  = 0.0;
    double search_ms = 0.0;
    double merge_ms  = 0.0;
    double final_ms  = 0.0;
    int clusters     = 0;
};

struct PointIndex_NumberTag_FEC {
    int nPointIndex = 0;
    int nNumberTag  = 0;
};

inline bool NumberTag_FEC(const PointIndex_NumberTag_FEC& p0,
                          const PointIndex_NumberTag_FEC& p1) {
    return p0.nNumberTag < p1.nNumberTag;
}

inline void ResetStats(FECStageStats* stats) {
    if (stats) *stats = FECStageStats{};
}

// Original FEC logic: KD-tree radius search + global scan relabel merge.
// Added optional stats pointer; original 4-parameter call remains valid.
inline std::vector<pcl::PointIndices> FEC(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_component_size,
    double tolorance,
    int max_n,
    FECStageStats* stats = nullptr
) {
    using Clock = std::chrono::steady_clock;
    ResetStats(stats);

    auto t_total_begin = Clock::now();

    if (!cloud || cloud->size() < static_cast<std::size_t>(min_component_size)) {
        return {};
    }

    double build_ms = 0.0;
    double search_ms = 0.0;
    double merge_ms = 0.0;
    double final_ms = 0.0;

    auto tb0 = Clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtreeflann;
    cloud_kdtreeflann.setInputCloud(cloud);
    auto tb1 = Clock::now();
    build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

    const int cloud_size = static_cast<int>(cloud->size());
    std::vector<int> marked_indices(cloud_size, 0);

    std::vector<int> pointIdx;
    std::vector<float> pointSquaredDistance;
    pointIdx.reserve(max_n > 0 ? std::min(max_n, cloud_size) : cloud_size);
    pointSquaredDistance.reserve(max_n > 0 ? std::min(max_n, cloud_size) : cloud_size);

    int tag_num = 1;

    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] == 0) {
            pointIdx.clear();
            pointSquaredDistance.clear();

            auto ts0 = Clock::now();
            cloud_kdtreeflann.radiusSearch(
                cloud->points[i], tolorance, pointIdx, pointSquaredDistance, max_n
            );
            auto ts1 = Clock::now();
            search_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

            int min_tag_num = tag_num;
            for (int idx : pointIdx) {
                if (marked_indices[idx] > 0 && marked_indices[idx] < min_tag_num) {
                    min_tag_num = marked_indices[idx];
                }
            }

            auto tm0 = Clock::now();
            for (int idx : pointIdx) {
                const int temp_tag_num = marked_indices[idx];

                // FEC original global scan relabel. This is intentionally inside merge timing.
                if (temp_tag_num > min_tag_num) {
                    for (int k = 0; k < cloud_size; ++k) {
                        if (marked_indices[k] == temp_tag_num) {
                            marked_indices[k] = min_tag_num;
                        }
                    }
                }
                marked_indices[idx] = min_tag_num;
            }
            auto tm1 = Clock::now();
            merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();

            ++tag_num;
        }
    }

    auto tf0 = Clock::now();

    std::vector<PointIndex_NumberTag_FEC> indices_tags(cloud_size);
    for (int i = 0; i < cloud_size; ++i) {
        indices_tags[i].nPointIndex = i;
        indices_tags[i].nNumberTag = marked_indices[i];
    }

    std::sort(indices_tags.begin(), indices_tags.end(), NumberTag_FEC);

    std::vector<pcl::PointIndices> cluster_indices;
    int begin_index = 0;
    for (int i = 0; i < cloud_size; ++i) {
        if (indices_tags[i].nNumberTag != indices_tags[begin_index].nNumberTag) {
            const int cluster_size = i - begin_index;
            if (cluster_size >= min_component_size) {
                pcl::PointIndices inliers;
                inliers.indices.reserve(cluster_size);
                for (int j = begin_index; j < i; ++j) {
                    inliers.indices.push_back(indices_tags[j].nPointIndex);
                }
                cluster_indices.push_back(std::move(inliers));
            }
            begin_index = i;
        }
    }

    const int final_cluster_size = cloud_size - begin_index;
    if (final_cluster_size >= min_component_size) {
        pcl::PointIndices inliers;
        inliers.indices.reserve(final_cluster_size);
        for (int j = begin_index; j < cloud_size; ++j) {
            inliers.indices.push_back(indices_tags[j].nPointIndex);
        }
        cluster_indices.push_back(std::move(inliers));
    }

    auto tf1 = Clock::now();
    final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

    auto t_total_end = Clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();

    if (stats) {
        stats->total_ms = total_ms;
        stats->build_ms = build_ms;
        stats->search_ms = search_ms;
        stats->merge_ms = merge_ms;
        stats->final_ms = final_ms;
        stats->clusters = static_cast<int>(cluster_indices.size());
    }

    return cluster_indices;
}

#endif // PCL_SEGEMENT_FEC_TIMED_H
