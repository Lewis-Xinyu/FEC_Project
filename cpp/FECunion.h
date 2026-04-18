#pragma once
#ifndef PCL_SEGEMENT_FECUNION_H
#define PCL_SEGEMENT_FECUNION_H

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/PointIndices.h>

#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

struct LabelUnionFind_FEC1 {
    std::vector<int> parent;
    std::vector<int> min_label;

    explicit LabelUnionFind_FEC1(int n) : parent(n), min_label(n) {
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
            min_label[i] = i;
        }
    }

    int find(int x) {
        int root = x;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[x] != x) {
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    }

    void unite(int a, int b) {
        int root_a = find(a);
        int root_b = find(b);
        if (root_a != root_b) {
            if (min_label[root_a] < min_label[root_b]) {
                parent[root_b] = root_a;
            } else {
                parent[root_a] = root_b;
                min_label[root_b] = std::min(min_label[root_a], min_label[root_b]);
            }
        }
    }

    int canonical_label(int x) {
        return min_label[find(x)];
    }
};

inline std::vector<pcl::PointIndices> FECunion(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_component_size,
    double tolorance,
    int max_n
) {
    using Clock = std::chrono::high_resolution_clock;

    auto t_total_begin = Clock::now();
    double build_ms = 0.0, search_ms = 0.0, merge_ms = 0.0, final_ms = 0.0;

    if (cloud->size() < static_cast<std::size_t>(min_component_size)) {
        return {};
    }

    auto t0 = Clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtreeflann;
    cloud_kdtreeflann.setInputCloud(cloud);
    auto t1 = Clock::now();
    build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int cloud_size = static_cast<int>(cloud->size());
    std::vector<int> marked_indices(cloud_size, 0);

    std::vector<int> pointIdx;
    std::vector<float> pointSquaredDistance;
    pointIdx.reserve(max_n);
    pointSquaredDistance.reserve(max_n);

    int tag_num = 1;
    LabelUnionFind_FEC1 tag_uf(cloud_size + 1);

    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] == 0) {
            pointIdx.clear();
            pointSquaredDistance.clear();

            auto ts0 = Clock::now();
            cloud_kdtreeflann.radiusSearch(cloud->points[i], tolorance, pointIdx, pointSquaredDistance, max_n);
            auto ts1 = Clock::now();
            search_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

            int min_tag_num = tag_num;
            for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
                int idx = pointIdx[j];
                if (marked_indices[idx] > 0) {
                    int current_tag = tag_uf.canonical_label(marked_indices[idx]);
                    if (current_tag < min_tag_num) {
                        min_tag_num = current_tag;
                    }
                }
            }

            if (min_tag_num == tag_num) {
                tag_num++;
            }

            auto tm0 = Clock::now();
            for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
                int idx = pointIdx[j];
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
            label_counts[marked_indices[i]]++;
        }
    }

    std::vector<int> label_to_cluster_id(tag_num, -1);
    std::vector<pcl::PointIndices> cluster_indices;

    for (int label = 1; label < tag_num; ++label) {
        if (label_counts[label] >= min_component_size) {
            label_to_cluster_id[label] = static_cast<int>(cluster_indices.size());
            pcl::PointIndices inliers;
            inliers.indices.reserve(label_counts[label]);
            cluster_indices.push_back(std::move(inliers));
        }
    }

    for (int i = 0; i < cloud_size; ++i) {
        int label = marked_indices[i];
        if (label > 0) {
            int cluster_id = label_to_cluster_id[label];
            if (cluster_id != -1) {
                cluster_indices[cluster_id].indices.push_back(i);
            }
        }
    }

    auto tf1 = Clock::now();
    final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

    auto t_total_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();

    std::cout << std::fixed << std::setprecision(3)
              << "FECunion total= " << total_ms << " ms "
              << "build= " << build_ms << " ms "
              << "search= " << search_ms << " ms "
              << "merge= " << merge_ms << " ms "
              << "final= " << final_ms << " ms "
              << "clusters=" << cluster_indices.size() << "\n";

    return cluster_indices;
}

#endif
