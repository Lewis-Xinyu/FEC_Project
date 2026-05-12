#pragma once

#include <algorithm>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fec_benchmark {

struct UnionFindLabels {
    std::vector<int> parent;
    std::vector<int> min_label;

    explicit UnionFindLabels(int n) : parent(n), min_label(n) {
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

inline std::vector<pcl::PointIndices> run_fecunion(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    int min_component_size,
    double tolerance,
    int max_n
) {
    if (cloud->size() < static_cast<std::size_t>(min_component_size)) {
        return {};
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    const int cloud_size = static_cast<int>(cloud->size());
    std::vector<int> marked_indices(cloud_size, 0);
    std::vector<int> point_indices;
    std::vector<float> point_squared_distance;
    if (max_n > 0) {
        point_indices.reserve(max_n);
        point_squared_distance.reserve(max_n);
    }

    int tag_num = 1;
    UnionFindLabels uf(cloud_size + 1);

    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] != 0) continue;

        point_indices.clear();
        point_squared_distance.clear();
        kdtree.radiusSearch(cloud->points[i], tolerance, point_indices, point_squared_distance, max_n);

        int min_tag_num = tag_num;
        for (int j = 0; j < static_cast<int>(point_indices.size()); ++j) {
            const int idx = point_indices[j];
            if (marked_indices[idx] > 0) {
                const int current_tag = uf.canonical_label(marked_indices[idx]);
                if (current_tag < min_tag_num) {
                    min_tag_num = current_tag;
                }
            }
        }

        if (min_tag_num == tag_num) {
            ++tag_num;
        }

        for (int j = 0; j < static_cast<int>(point_indices.size()); ++j) {
            const int idx = point_indices[j];
            if (marked_indices[idx] > 0) {
                uf.unite(marked_indices[idx], min_tag_num);
            } else {
                marked_indices[idx] = min_tag_num;
            }
        }
    }

    std::vector<int> label_counts(tag_num, 0);
    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] > 0) {
            marked_indices[i] = uf.canonical_label(marked_indices[i]);
            ++label_counts[marked_indices[i]];
        }
    }

    std::vector<int> label_to_cluster_id(tag_num, -1);
    std::vector<pcl::PointIndices> clusters;
    for (int label = 1; label < tag_num; ++label) {
        if (label_counts[label] >= min_component_size) {
            label_to_cluster_id[label] = static_cast<int>(clusters.size());
            pcl::PointIndices inliers;
            inliers.indices.reserve(label_counts[label]);
            clusters.push_back(std::move(inliers));
        }
    }

    for (int i = 0; i < cloud_size; ++i) {
        const int label = marked_indices[i];
        if (label > 0) {
            const int cluster_id = label_to_cluster_id[label];
            if (cluster_id != -1) {
                clusters[cluster_id].indices.push_back(i);
            }
        }
    }

    return clusters;
}

}  // namespace fec_benchmark
