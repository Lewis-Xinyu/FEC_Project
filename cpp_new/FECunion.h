#pragma once
#ifndef PCL_SEGEMENT_FECUNION_TIMED_H
#define PCL_SEGEMENT_FECUNION_TIMED_H

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/PointIndices.h>

#include <vector>
#include <algorithm>
#include <chrono>
#include <cstddef>

struct LabelUnionFind_FECunion_Timed {
    std::vector<int> parent;
    std::vector<int> min_label;

    explicit LabelUnionFind_FECunion_Timed(int n) : parent(n), min_label(n) {
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            min_label[i] = i;
        }
    }

    int find(int x) {
        int root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != x) {
            const int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    }

    void unite(int a, int b) {
        const int root_a = find(a);
        const int root_b = find(b);
        if (root_a == root_b) return;

        if (min_label[root_a] <= min_label[root_b]) {
            parent[root_b] = root_a;
            min_label[root_a] = std::min(min_label[root_a], min_label[root_b]);
        } else {
            parent[root_a] = root_b;
            min_label[root_b] = std::min(min_label[root_a], min_label[root_b]);
        }
    }

    int canonical_label(int x) {
        return min_label[find(x)];
    }
};

struct FECunionStageStats {
    double total_ms  = 0.0;
    double build_ms  = 0.0;
    double search_ms = 0.0;
    double merge_ms  = 0.0;
    double final_ms  = 0.0;
    int clusters     = 0;
};

inline void ResetStats(FECunionStageStats* stats) {
    if (stats) *stats = FECunionStageStats{};
}

// FECunion: KD-tree radius search + union-find label merge.
// Added optional stats pointer; original 4-parameter call remains valid.
inline std::vector<pcl::PointIndices> FECunion(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_component_size,
    double tolorance,
    int max_n,
    FECunionStageStats* stats = nullptr
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
    LabelUnionFind_FECunion_Timed tag_uf(cloud_size + 1);

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
                if (marked_indices[idx] > 0) {
                    const int current_tag = tag_uf.canonical_label(marked_indices[idx]);
                    if (current_tag < min_tag_num) min_tag_num = current_tag;
                }
            }

            if (min_tag_num == tag_num) ++tag_num;

            auto tm0 = Clock::now();
            for (int idx : pointIdx) {
                if (marked_indices[idx] > 0) {
                    tag_uf.unite(marked_indices[idx], min_tag_num);
                } else {
                    marked_indices[idx] = min_tag_num;
                }
            }
            auto tm1 = Clock::now();
            merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();
        }
    }

    auto tf0 = Clock::now();

    std::vector<int> label_counts(tag_num, 0);
    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] > 0) {
            marked_indices[i] = tag_uf.canonical_label(marked_indices[i]);
            ++label_counts[marked_indices[i]];
        }
    }

    std::vector<int> label_to_cluster_id(tag_num, -1);
    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices.reserve(tag_num > 1 ? tag_num - 1 : 0);

    for (int label = 1; label < tag_num; ++label) {
        if (label_counts[label] >= min_component_size) {
            label_to_cluster_id[label] = static_cast<int>(cluster_indices.size());
            pcl::PointIndices inliers;
            inliers.indices.reserve(label_counts[label]);
            cluster_indices.push_back(std::move(inliers));
        }
    }

    for (int i = 0; i < cloud_size; ++i) {
        const int label = marked_indices[i];
        if (label > 0) {
            const int cluster_id = label_to_cluster_id[label];
            if (cluster_id != -1) {
                cluster_indices[cluster_id].indices.push_back(i);
            }
        }
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

#endif // PCL_SEGEMENT_FECUNION_TIMED_H
