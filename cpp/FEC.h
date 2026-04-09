#pragma once
#pragma warning(disable:4996)
#ifndef PCL_SEGEMENT_FEC_H
#define PCL_SEGEMENT_FEC_H

#endif //PCL_SEGEMENT_FEC_H

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <vector>
#include <pcl/io/ply_io.h>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <omp.h>
using namespace std;

/**
* Store index and label information for each point
*/
struct PointIndex_NumberTag
{
    float nPointIndex;
    float nNumberTag;
};

bool NumberTag(const PointIndex_NumberTag& p0, const PointIndex_NumberTag& p1)
{
    return p0.nNumberTag < p1.nNumberTag;
}

std::vector<pcl::PointIndices> FEC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int min_component_size, double tolorance, int max_n) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_total_begin = Clock::now();

    unsigned long i, j;
    if (cloud->size() < min_component_size)
    {
        PCL_ERROR("Could not find any cluster");
    }

    auto tb0 = Clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZ>cloud_kdtreeflann;
    cloud_kdtreeflann.setInputCloud(cloud);
    auto tb1 = Clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

    int cloud_size = cloud->size();
    std::vector<int> marked_indices;
    marked_indices.resize(cloud_size);


    memset(marked_indices.data(), 0, sizeof(int) * cloud_size);
    std::vector<int> pointIdx;
    std::vector<float> pointquaredDistance;

    int tag_num = 1, temp_tag_num = -1;
    double search_ms = 0.0;
    double merge_ms = 0.0;

    for (i = 0; i < cloud_size; i++)
    {
        // Clustering process
        if (marked_indices[i] == 0) // reset to initial value if this point has not been manipulated
        {
            pointIdx.clear();
            pointquaredDistance.clear();
            auto ts0 = Clock::now();
            cloud_kdtreeflann.radiusSearch(cloud->points[i], tolorance, pointIdx, pointquaredDistance, max_n);
            auto ts1 = Clock::now();
            search_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();
            /**
            * All neighbors closest to a specified point with a query within a given radius
            * para.tolorance is the radius of the sphere that surrounds all neighbors
            * pointIdx is the resulting index of neighboring points
            * pointquaredDistance is the final square distance to adjacent points
            * pointIdx.size() is the maximum number of neighbors returned by limit
            */
            int min_tag_num = tag_num;
            for (j = 0; j < pointIdx.size(); j++)
            {
                /**
                 * find the minimum label value contained in the field points, and tag it to this cluster label.
                 */
                if ((marked_indices[pointIdx[j]] > 0) && (marked_indices[pointIdx[j]] < min_tag_num))
                {
                    min_tag_num = marked_indices[pointIdx[j]];
                }
            }
            auto tm0 = Clock::now();
            for (j = 0; j < pointIdx.size(); j++)
            {
                temp_tag_num = marked_indices[pointIdx[j]];

                /*
                 * Each domain point, as well as all points in the same cluster, is uniformly assigned this label
                 */
                if (temp_tag_num > min_tag_num)
                {
                    for (int k = 0; k < cloud_size; k++)
                    {
                        if (marked_indices[k] == temp_tag_num)
                        {
                            marked_indices[k] = min_tag_num;
                        }
                    }
                }
                marked_indices[pointIdx[j]] = min_tag_num;
            }
            auto tm1 = Clock::now();
            merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();
            tag_num++;
        }
    }

    auto tf0 = Clock::now();
    std::vector<PointIndex_NumberTag> indices_tags;
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    indices_tags.resize(cloud_size);

    PointIndex_NumberTag temp_index_tag;


    for (i = 0; i < cloud_size; i++)
    {
        /**
        * Put each point index and the corresponding tag value into the indices_tags
        */
        temp_index_tag.nPointIndex = i;
        temp_index_tag.nNumberTag = marked_indices[i];

        indices_tags[i] = temp_index_tag;
    }

    sort(indices_tags.begin(), indices_tags.end(), NumberTag);

    unsigned long begin_index = 0;
    for (i = 0; i < indices_tags.size(); i++)
    {
        // Relabel each cluster
        if (indices_tags[i].nNumberTag != indices_tags[begin_index].nNumberTag)
        {
            if ((i - begin_index) >= min_component_size)
            {
                unsigned long m = 0;
                inliers->indices.resize(i - begin_index);
                for (j = begin_index; j < i; j++)
                    inliers->indices[m++] = indices_tags[j].nPointIndex;
                cluster_indices.push_back(*inliers);
            }
            begin_index = i;
        }
    }

    if ((i - begin_index) >= min_component_size)
    {
        for (j = begin_index; j < i; j++)
        {
            unsigned long m = 0;
            inliers->indices.resize(i - begin_index);
            for (j = begin_index; j < i; j++)
            {
                inliers->indices[m++] = indices_tags[j].nPointIndex;
            }
            cluster_indices.push_back(*inliers);
        }
    }
    auto tf1 = Clock::now();
    const double final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();
    const double total_ms = std::chrono::duration<double, std::milli>(tf1 - t_total_begin).count();

    std::cout << std::fixed << std::setprecision(3)
              << "FEC total= " << total_ms << " ms "
              << "build= " << build_ms << " ms "
              << "search= " << search_ms << " ms "
              << "merge= " << merge_ms << " ms "
              << "final= " << final_ms << " ms "
              << "clusters=" << cluster_indices.size() << "\n";

    return cluster_indices;

}
