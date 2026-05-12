#pragma once

#include <algorithm>
#include <chrono>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "FEC_profile.h"

struct FECUnionProfileUnionFind {
    std::vector<int> parent;
    std::vector<int> min_label;

    explicit FECUnionProfileUnionFind(int n) : parent(n), min_label(n) {
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
        if (min_label[root_a] < min_label[root_b]) {
            parent[root_b] = root_a;
        } else {
            parent[root_a] = root_b;
            min_label[root_b] = std::min(min_label[root_a], min_label[root_b]);
        }
    }

    int canonical_label(int x) {
        return min_label[find(x)];
    }
};

inline FECProfileResult FECunion_profile(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    int min_component_size,
    double tolerance,
    int max_n
) {
    using Clock = std::chrono::steady_clock;

    FECProfileResult result;
    result.stats.points = static_cast<int>(cloud->size());
    result.stats.min_cluster_size = min_component_size;
    result.stats.tolerance = tolerance;
    result.stats.max_n = max_n;

    if (cloud->size() < static_cast<std::size_t>(min_component_size)) {
        return result;
    }

    const auto t_total_begin = Clock::now();

    const auto tb0 = Clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtreeflann;
    cloud_kdtreeflann.setInputCloud(cloud);
    const auto tb1 = Clock::now();
    result.stats.build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

    const int cloud_size = static_cast<int>(cloud->size());
    std::vector<int> marked_indices(cloud_size, 0);

    std::vector<int> pointIdx;
    std::vector<float> pointSquaredDistance;
    if (max_n > 0) {
        pointIdx.reserve(max_n);
        pointSquaredDistance.reserve(max_n);
    }

    int tag_num = 1;
    FECUnionProfileUnionFind tag_uf(cloud_size + 1);

    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] != 0) continue;

        pointIdx.clear();
        pointSquaredDistance.clear();

        const auto ts0 = Clock::now();
        cloud_kdtreeflann.radiusSearch(cloud->points[i], tolerance, pointIdx, pointSquaredDistance, max_n);
        const auto ts1 = Clock::now();

        ++result.stats.search_calls;
        result.stats.search_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
        result.stats.total_neighbors_returned += static_cast<std::uint64_t>(pointIdx.size());
        result.stats.max_neighbors_returned =
            std::max(result.stats.max_neighbors_returned, static_cast<int>(pointIdx.size()));
        if (max_n > 0 && static_cast<int>(pointIdx.size()) == max_n) {
            ++result.stats.saturation_count;
        }

        int min_tag_num = tag_num;
        result.stats.neighbor_scan_pass1 += static_cast<std::uint64_t>(pointIdx.size());
        for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
            const int idx = pointIdx[j];
            if (marked_indices[idx] > 0) {
                const int current_tag = tag_uf.canonical_label(marked_indices[idx]);
                if (current_tag < min_tag_num) min_tag_num = current_tag;
            }
        }

        if (min_tag_num == tag_num) ++tag_num;

        const auto tm0 = Clock::now();
        result.stats.neighbor_scan_pass2 += static_cast<std::uint64_t>(pointIdx.size());
        for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
            const int idx = pointIdx[j];
            if (marked_indices[idx] > 0) {
                tag_uf.unite(marked_indices[idx], min_tag_num);
            } else {
                marked_indices[idx] = min_tag_num;
            }
        }
        const auto tm1 = Clock::now();
        result.stats.merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();
    }

    const auto tf0 = Clock::now();

    std::vector<int> label_counts(tag_num, 0);
    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] > 0) {
            marked_indices[i] = tag_uf.canonical_label(marked_indices[i]);
            label_counts[marked_indices[i]]++;
        }
    }

    std::vector<int> label_to_cluster_id(tag_num, -1);
    for (int label = 1; label < tag_num; ++label) {
        if (label_counts[label] >= min_component_size) {
            label_to_cluster_id[label] = static_cast<int>(result.clusters.size());
            pcl::PointIndices inliers;
            inliers.indices.reserve(label_counts[label]);
            result.clusters.push_back(std::move(inliers));
        }
    }

    for (int i = 0; i < cloud_size; ++i) {
        const int label = marked_indices[i];
        if (label > 0) {
            const int cluster_id = label_to_cluster_id[label];
            if (cluster_id != -1) {
                result.clusters[cluster_id].indices.push_back(i);
            }
        }
    }

    const auto tf1 = Clock::now();
    result.stats.final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
    result.stats.total_ms = std::chrono::duration<double, std::milli>(tf1 - t_total_begin).count();
    result.stats.clusters = static_cast<int>(result.clusters.size());
    for (const auto& cluster : result.clusters) {
        result.stats.clustered_points += static_cast<int>(cluster.indices.size());
    }
    return result;
}
