#pragma once
#ifndef PCL_SEGEMENT_FEC1_TIMED_H
#define PCL_SEGEMENT_FEC1_TIMED_H

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/PointIndices.h>

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <cstddef>

struct FEC1StageStats {
    double total_ms  = 0.0;
    double build_ms  = 0.0;
    double search_ms = 0.0;
    double merge_ms  = 0.0;
    double final_ms  = 0.0;
    int clusters     = 0;
};

struct PointIndex_Tag_FEC1_Timed {
    int nPointIndex = 0;
    int nNumberTag  = 0;
};

inline bool NumberTag_FEC1_Timed(const PointIndex_Tag_FEC1_Timed& p0,
                                 const PointIndex_Tag_FEC1_Timed& p1) {
    return p0.nNumberTag < p1.nNumberTag;
}

inline void ResetStats(FEC1StageStats* stats) {
    if (stats) *stats = FEC1StageStats{};
}

// FEC1/FEC++ style: KD-tree radius search + hash-based inverted index relabel.
// Added optional stats pointer; original 4-parameter call remains valid.
inline std::vector<pcl::PointIndices> FEC1(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_component_size,
    double tolorance,
    int max_n,
    FEC1StageStats* stats = nullptr
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
    std::unordered_map<int, std::vector<int>> tag_to_indices;
    tag_to_indices.reserve(static_cast<std::size_t>(cloud_size / 8 + 16));

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
                const int current_tag = marked_indices[idx];

                if (current_tag > 0 && current_tag != min_tag_num) {
                    auto it = tag_to_indices.find(current_tag);
                    if (it != tag_to_indices.end()) {
                        auto& points = it->second;
                        auto& target = tag_to_indices[min_tag_num];
                        target.reserve(target.size() + points.size());
                        for (int p : points) {
                            marked_indices[p] = min_tag_num;
                            target.push_back(p);
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

    std::vector<PointIndex_Tag_FEC1_Timed> indices_tags(cloud_size);
    for (int i = 0; i < cloud_size; ++i) {
        indices_tags[i].nPointIndex = i;
        indices_tags[i].nNumberTag = marked_indices[i];
    }

    std::sort(indices_tags.begin(), indices_tags.end(), NumberTag_FEC1_Timed);

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

#endif // PCL_SEGEMENT_FEC1_TIMED_H
