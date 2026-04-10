#pragma warning(disable:4996)
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

// 引入你的所有算法头文件
#include "FEC.h"
#include "FEC_Block.h"
#include "FEC1.h"
#include "FEC1_1.h"
#include "EC.h"
#include "EC_block.h"
#include "RG.h"
#include "FEC_Union.h"
#include "FEC_Union_Block.h"
#include "FEC_Union_Grid_Block.h"
#include "FEC1_improved_block_fixed.h"
#include "Voxel_FEC1.h"
using namespace std;

// ==========================================
// 辅助函数：负责把聚类结果画出来 (封装起来，保持 main 干净)
// ==========================================
void visualize_clusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                        const std::vector<pcl::PointIndices>& cluster_indices, 
                        const std::string& window_name) 
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_name));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_point(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 为每个聚类分配随机颜色
    srand(static_cast<unsigned int>(time(0))); 
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        uint8_t r = rand() % 256;
        uint8_t g = rand() % 256;
        uint8_t b = rand() % 256;

        for (size_t j = 0; j < cluster_indices[i].indices.size(); j++) {
            pcl::PointXYZRGB point;
            int idx = cluster_indices[i].indices[j];
            point.x = cloud->points[idx].x;
            point.y = cloud->points[idx].y;
            point.z = cloud->points[idx].z;
            point.r = r;
            point.g = g;
            point.b = b;
            color_point->push_back(point);
        }
    }

    viewer->addPointCloud(color_point, "colored_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "colored_cloud");
    viewer->spin();
}

// ==========================================
// 主函数：控制中心
// ==========================================
int main() {
    // 模式开关：
    // run_all_datasets   = true  时，按 datasets 列表依次运行所有数据集
    // run_all_algorithms = true  时，在当前数据集上依次运行所有算法
    // 两者都为 false 时，只运行 single_dataset_path + single_algorithm
    const bool run_all_algorithms = false;
    const bool enable_visualization = true;
    const bool run_all_datasets = false;

    // 单算法模式下可选算法：
    // "FEC"              - 最早的 FEC，原始标签传播版本
    // "FEC1"             - 使用哈希表优化标签合并
    // "FEC1_1"           - FEC1_1 版本
    // "FEC_Union"        - 使用并查集替代标签/哈希表合并
    // "EC"               - PCL 原生 EuclideanClusterExtraction
    // "RG"               - Region Growing
    // "FEC_Block"        - FEC 的分块并行版本
    // "FEC1_Block"       - FEC1 的分块并行版本
    // "FEC_Union_Block"  - FEC_Union 的分块并行版本
    // "FEC_Union_Grid_Block" - FEC_Union 的网格哈希分块并行版本
    // "Voxel_FEC1"       - 基于体素哈希搜索的 FEC1 版本
    // "EC_Block"         - EC 的分块并行版本
    // 单算法模式下，直接修改这两个字符串即可：
    // single_dataset_path 选择数据集
    // single_algorithm    选择算法
    const std::string single_algorithm = "Voxel_FEC1";
    const std::string single_dataset_path = "./data/046.ply";
    // 通用参数
    int min_cluster_size = 100;
    double tolerance = 0.2; 
    int max_n = 50;

    // RG (区域生长) 专属参数
    float smoothness_threshold = 3.0; // 角度
    float curvature_threshold = 1.0;  // 曲率
    // --------------------------------------------------

    const std::vector<std::string> algorithms = {
        "FEC",
        "FEC1",
        "FEC1_1",
        "FEC_Union",
        "EC",
        "RG",
        "FEC_Block",
        "FEC1_Block",
        "FEC_Union_Block",
        "FEC_Union_Grid_Block",
        "Voxel_FEC1",
        "EC_Block"
    };

    const std::vector<std::string> datasets = {
        "./data/001.ply",
        "./data/015.ply",
        "./data/028.ply",
        "./data/046.ply",
        "./data/066.ply",
        "./data/084.ply",
        "./data/car.ply",
        "./data/street.ply"
    };

    auto run_algorithm = [&](pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                             const std::string& current_algorithm) -> std::vector<pcl::PointIndices> {
        if (current_algorithm == "FEC") {
            return FEC(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC_Block") {
            return FEC_Block(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC1") {
            return FEC1(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC1_1") {
            return FEC1_1(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "EC") {
            return EC(cloud, tolerance, min_cluster_size);
        }
        if (current_algorithm == "EC_Block") {
            return EC_Block(cloud, tolerance, min_cluster_size);
        }
        if (current_algorithm == "RG") {
            return RG(cloud, min_cluster_size, max_n, smoothness_threshold, curvature_threshold);
        }
        if (current_algorithm == "FEC_Union") {
            return FEC_Union(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC_Union_Block") {
            return FEC_Union_Block(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC_Union_Grid_Block") {
            return FEC_Union_Grid_Block(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "Voxel_FEC1") {
            return Voxel_FEC1(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC1_Block") {
            return FEC1_Block(cloud, min_cluster_size, tolerance, max_n);
        }

        cout << "Unknown algorithm: " << current_algorithm << endl;
        return {};
    };

    auto load_cloud = [&](const std::string& dataset_path) -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PLYReader readerPLY;
        if (readerPLY.read(dataset_path, *cloud) == -1) {
            PCL_ERROR("Failed to read point cloud file.\n");
            return nullptr;
        }
        return cloud;
    };

    if (run_all_datasets) {
        for (const auto& dataset_path : datasets) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_cloud(dataset_path);
            if (!cloud) {
                cout << "Dataset: " << dataset_path << " load failed." << endl;
                continue;
            }

            const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();
            cout << "\nDataset: " << dataset_name << " Points: " << cloud->size() << endl;
            cout << left << setw(24) << "Algorithm" << "Clusters" << endl;
            cout << left << setw(24) << "---------" << "--------" << endl;

            for (const auto& algorithm : algorithms) {
                std::vector<pcl::PointIndices> cluster_indices = run_algorithm(cloud, algorithm);
                cout << left << setw(24) << algorithm << cluster_indices.size() << endl;
            }
        }
        return 0;
    }

    // 1. 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_cloud(single_dataset_path);
    if (!cloud) {
        return -1;
    }
    cout << "Loaded points: " << cloud->size() << endl;

    if (run_all_algorithms) {
        cout << left << setw(24) << "Algorithm" << "Clusters" << endl;
        cout << left << setw(24) << "---------" << "--------" << endl;
        for (const auto& algorithm : algorithms) {
            std::vector<pcl::PointIndices> cluster_indices = run_algorithm(cloud, algorithm);
            cout << left << setw(24) << algorithm << cluster_indices.size() << endl;
        }
        return 0;
    }

    std::vector<pcl::PointIndices> cluster_indices = run_algorithm(cloud, single_algorithm);
    cout << "Algorithm: " << single_algorithm << endl;
    cout << "Clusters: " << cluster_indices.size() << endl;

    if (enable_visualization) {
        visualize_clusters(cloud, cluster_indices, "Algorithm Test: " + single_algorithm);
    }

    return 0;
}
