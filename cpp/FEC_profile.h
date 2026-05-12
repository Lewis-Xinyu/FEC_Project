#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct FECProfilePointTag {
    float point_index = 0.0f;
    float label = 0.0f;
};

inline bool FECProfileLabelLess(const FECProfilePointTag& a, const FECProfilePointTag& b) {
    return a.label < b.label;
}

struct FECProfileStats {
    int points = 0;
    int min_cluster_size = 0;
    double tolerance = 0.0;
    int max_n = 0;

    int search_calls = 0;
    std::uint64_t total_neighbors_returned = 0;
    int max_neighbors_returned = 0;
    int saturation_count = 0;

    std::uint64_t neighbor_scan_pass1 = 0;
    std::uint64_t neighbor_scan_pass2 = 0;
    int relabel_trigger_count = 0;
    std::uint64_t relabel_point_visits = 0;

    double build_ms = 0.0;
    double search_ms = 0.0;
    double merge_ms = 0.0;
    double final_ms = 0.0;
    double total_ms = 0.0;

    int clusters = 0;
    int clustered_points = 0;
};

struct FECProfileResult {
    std::vector<pcl::PointIndices> clusters;
    FECProfileStats stats;
};

inline FECProfileResult FEC_profile(
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
    int temp_tag_num = -1;

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
            if ((marked_indices[pointIdx[j]] > 0) && (marked_indices[pointIdx[j]] < min_tag_num)) {
                min_tag_num = marked_indices[pointIdx[j]];
            }
        }

        const auto tm0 = Clock::now();
        result.stats.neighbor_scan_pass2 += static_cast<std::uint64_t>(pointIdx.size());
        for (int j = 0; j < static_cast<int>(pointIdx.size()); ++j) {
            temp_tag_num = marked_indices[pointIdx[j]];
            if (temp_tag_num > min_tag_num) {
                ++result.stats.relabel_trigger_count;
                result.stats.relabel_point_visits += static_cast<std::uint64_t>(cloud_size);
                for (int k = 0; k < cloud_size; ++k) {
                    if (marked_indices[k] == temp_tag_num) {
                        marked_indices[k] = min_tag_num;
                    }
                }
            }
            marked_indices[pointIdx[j]] = min_tag_num;
        }
        const auto tm1 = Clock::now();
        result.stats.merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();
        ++tag_num;
    }

    const auto tf0 = Clock::now();
    std::vector<FECProfilePointTag> indices_tags(cloud_size);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    for (int i = 0; i < cloud_size; ++i) {
        indices_tags[i].point_index = static_cast<float>(i);
        indices_tags[i].label = static_cast<float>(marked_indices[i]);
    }

    std::sort(indices_tags.begin(), indices_tags.end(), FECProfileLabelLess);

    std::size_t begin_index = 0;
    std::size_t i = 0;
    for (i = 0; i < indices_tags.size(); ++i) {
        if (indices_tags[i].label != indices_tags[begin_index].label) {
            if ((i - begin_index) >= static_cast<std::size_t>(min_component_size)) {
                std::size_t m = 0;
                inliers->indices.resize(i - begin_index);
                for (std::size_t j = begin_index; j < i; ++j) {
                    inliers->indices[m++] = static_cast<int>(indices_tags[j].point_index);
                }
                result.clusters.push_back(*inliers);
            }
            begin_index = i;
        }
    }

    if ((i - begin_index) >= static_cast<std::size_t>(min_component_size)) {
        std::size_t j = begin_index;
        for (; j < i; ++j) {
            std::size_t m = 0;
            inliers->indices.resize(i - begin_index);
            for (j = begin_index; j < i; ++j) {
                inliers->indices[m++] = static_cast<int>(indices_tags[j].point_index);
            }
            result.clusters.push_back(*inliers);
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
