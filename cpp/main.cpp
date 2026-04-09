#pragma warning(disable:4996)
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

// 引入你的所有算法头文件
#include "FEC.h"
#include "FEC1.h"
#include "EC.h"
#include "RG.h"
#include "FEC_Union.h"

using namespace std;
using namespace chrono;

// ==========================================
// 辅助函数：负责把聚类结果画出来 (封装起来，保持 main 干净)
// ==========================================
void visualize_clusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                        const std::vector<pcl::PointIndices>& cluster_indices, 
                        const std::string& window_name) 
{
    cout << "📦 正在准备 3D 渲染..." << endl;
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_name));
    viewer->setBackgroundColor(0.05, 0.05, 0.05); // 深灰色背景更酷

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
    
    cout << "✨ 渲染完成！请查看弹出的 3D 窗口。" << endl;
    viewer->spin();
}

// ==========================================
// 主函数：控制中心
// ==========================================
int main() {
    std::string current_algorithm = "FEC_Union"; // 可选: "FEC", "FEC1", "EC", "RG","FEC_Union"
    
    // 通用参数
    int min_cluster_size = 100;
    double tolerance = 0.2; 
    int max_n = 50;

    // RG (区域生长) 专属参数
    float smoothness_threshold = 3.0; // 角度
    float curvature_threshold = 1.0;  // 曲率
    // --------------------------------------------------

    // 1. 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PLYReader readerPLY;
    
    cout << "⏳ 正在读取点云数据..." << endl;
    if (readerPLY.read("./data/street.ply", *cloud) == -1) {
        PCL_ERROR("❌ 找不到点云文件，请检查路径！\n");
        return -1;
    }
    cout << "✅ 成功读取点云，共有: " << cloud->size() << " 个点。" << endl;

    // 2. 选择算法并计时
    std::vector<pcl::PointIndices> cluster_indices;
    cout << "\n🚀 正在运行算法: [" << current_algorithm << "] ..." << endl;
    
    auto start = system_clock::now();

    // ======= 算法路由中心 =======
    if (current_algorithm == "FEC") {
        cluster_indices = FEC(cloud, min_cluster_size, tolerance, max_n);
    } 
    else if (current_algorithm == "FEC1") {
        cluster_indices = FEC1(cloud, min_cluster_size, tolerance, max_n);
    } 
    else if (current_algorithm == "EC") {
        cluster_indices = EC(cloud, tolerance, min_cluster_size);
    } 
    else if (current_algorithm == "RG") {
        cluster_indices = RG(cloud, min_cluster_size, max_n, smoothness_threshold, curvature_threshold);
    } 
    else if (current_algorithm == "FEC_Union"){
        cluster_indices = FEC_Union(cloud, min_cluster_size, tolerance, max_n);
    }
    else {
        cout << "❌ 未知的算法名称！" << endl;
        return -1;
    }
    // ============================

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double time_spent = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    
    // 3. 输出实验结果
    cout << "----------------------------------------" << endl;
    cout << "🎯 算法: " << current_algorithm << " 运行完毕！" << endl;
    cout << "⏱️ 耗时: " << time_spent << " 秒" << endl;
    cout << "📦 共找到: " << cluster_indices.size() << " 个有效聚类簇" << endl;
    cout << "----------------------------------------\n" << endl;

    // 4. 可视化
    visualize_clusters(cloud, cluster_indices, "Algorithm Test: " + current_algorithm);

    return 0;
}