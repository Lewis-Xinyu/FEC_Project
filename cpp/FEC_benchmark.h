#pragma once

#include <algorithm>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fec_benchmark {

struct PointLabelPair {
    float point_index = 0.0f;
    float label = 0.0f;
};

inline bool label_less(const PointLabelPair& a, const PointLabelPair& b) {
    return a.label < b.label;
}

inline std::vector<pcl::PointIndices> run_fec(
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
    int temp_tag_num = -1;

    for (int i = 0; i < cloud_size; ++i) {
        if (marked_indices[i] != 0) continue;

        point_indices.clear();
        point_squared_distance.clear();
        kdtree.radiusSearch(cloud->points[i], tolerance, point_indices, point_squared_distance, max_n);

        int min_tag_num = tag_num;
        for (int j = 0; j < static_cast<int>(point_indices.size()); ++j) {
            const int idx = point_indices[j];
            if ((marked_indices[idx] > 0) && (marked_indices[idx] < min_tag_num)) {
                min_tag_num = marked_indices[idx];
            }
        }

        for (int j = 0; j < static_cast<int>(point_indices.size()); ++j) {
            temp_tag_num = marked_indices[point_indices[j]];
            if (temp_tag_num > min_tag_num) {
                for (int k = 0; k < cloud_size; ++k) {
                    if (marked_indices[k] == temp_tag_num) {
                        marked_indices[k] = min_tag_num;
                    }
                }
            }
            marked_indices[point_indices[j]] = min_tag_num;
        }
        ++tag_num;
    }

    std::vector<PointLabelPair> labels(cloud_size);
    for (int i = 0; i < cloud_size; ++i) {
        labels[i].point_index = static_cast<float>(i);
        labels[i].label = static_cast<float>(marked_indices[i]);
    }
    std::sort(labels.begin(), labels.end(), label_less);

    std::vector<pcl::PointIndices> clusters;
    pcl::PointIndices current;

    std::size_t begin_index = 0;
    std::size_t i = 0;
    for (i = 0; i < labels.size(); ++i) {
        if (labels[i].label != labels[begin_index].label) {
            if ((i - begin_index) >= static_cast<std::size_t>(min_component_size)) {
                current.indices.resize(i - begin_index);
                std::size_t m = 0;
                for (std::size_t j = begin_index; j < i; ++j) {
                    current.indices[m++] = static_cast<int>(labels[j].point_index);
                }
                clusters.push_back(current);
            }
            begin_index = i;
        }
    }

    if ((i - begin_index) >= static_cast<std::size_t>(min_component_size)) {
        current.indices.resize(i - begin_index);
        std::size_t m = 0;
        for (std::size_t j = begin_index; j < i; ++j) {
            current.indices[m++] = static_cast<int>(labels[j].point_index);
        }
        clusters.push_back(current);
    }

    return clusters;
}

}  // namespace fec_benchmark
