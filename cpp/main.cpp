#pragma warning(disable:4996)
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <optional>
#include <algorithm>
#include <cmath>
#include <cctype>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "FEC.h"
#include "FEC_Block.h"
#include "FEC1.h"
#include "FEC1_1.h"
#include "EC.h"
#include "EC_block.h"
#include "RG.h"
#include "FECunion.h"
#include "FEC_Union.h"
#include "FEC_Union_Block.h"
#ifdef PCL_SEGEMENT_FEC_UNION_BLOCK_H
#undef PCL_SEGEMENT_FEC_UNION_BLOCK_H
#endif
#define FECUB_PointIndexTag FECUBN_PointIndexTag
#define FECUB_TagLess FECUBN_TagLess
#define FECUB_BlockKey FECUBN_BlockKey
#define FECUB_BlockKeyHash FECUBN_BlockKeyHash
#define FECUB_LocalCluster FECUBN_LocalCluster
#define FECUB_LocalStats FECUBN_LocalStats
#define FECUB_BlockData FECUBN_BlockData
#define FECUB_DisjointSet FECUBN_DisjointSet
#define FECUB_LocalClusterOnly FECUBN_LocalClusterOnly
#define FEC_Union_Block FEC_Union_Block_new
#include "FEC_Union_Block_new.h"
#undef FECUB_PointIndexTag
#undef FECUB_TagLess
#undef FECUB_BlockKey
#undef FECUB_BlockKeyHash
#undef FECUB_LocalCluster
#undef FECUB_LocalStats
#undef FECUB_BlockData
#undef FECUB_DisjointSet
#undef FECUB_LocalClusterOnly
#undef FEC_Union_Block
#include "FEC_Union_Block_new2.h"
#include "FEC_Union_Grid_Block.h"
#include "FECunion_Block.h"
#include "FEC1_improved_block_fixed.h"
#include "Voxel_FEC1.h"
using namespace std;

struct RunMetrics {
    double total_ms = 0.0;
    double build_ms = 0.0;
    double search_ms = 0.0;
    double merge_ms = 0.0;
    double final_ms = 0.0;
    int clusters = 0;
};

static std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::size_t begin = 0;
        while (begin < item.size() && std::isspace(static_cast<unsigned char>(item[begin]))) {
            ++begin;
        }
        std::size_t end = item.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(item[end - 1]))) {
            --end;
        }
        if (end > begin) {
            parts.push_back(item.substr(begin, end - begin));
        }
    }
    return parts;
}

static bool parse_bool_arg(const std::string& value, bool fallback) {
    std::string lower;
    lower.reserve(value.size());
    for (char ch : value) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
    return fallback;
}

static std::optional<double> extract_metric(const std::string& text, const std::string& key) {
    const std::string token = key + "= ";
    std::size_t pos = text.find(token);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos += token.size();
    std::size_t end = text.find_first_of(" \n\r", pos);
    const std::string value = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    try {
        return std::stod(value);
    } catch (...) {
        return std::nullopt;
    }
}

static std::optional<int> extract_cluster_count(const std::string& text) {
    const std::string token = "clusters=";
    std::size_t pos = text.rfind(token);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos += token.size();
    std::size_t end = text.find_first_of(" \n\r", pos);
    const std::string value = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    try {
        return std::stoi(value);
    } catch (...) {
        return std::nullopt;
    }
}

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
int main(int argc, char** argv) {
    // 模式开关：
    // run_all_datasets   = true  时，按 datasets 列表依次运行所有数据集
    // run_all_algorithms = true  时，在当前数据集上依次运行所有算法
    // 两者都为 false 时，只运行 single_dataset_path + single_algorithm
    const bool run_all_algorithms = false;
    const bool enable_visualization = false;
    const bool run_all_datasets = false;
    const int repeat_runs = 7;
    const int warmup_runs = 1;

    // 单算法模式下可选算法：
    // "FEC"              - 最早的 FEC，原始标签传播版本
    // "FEC1"             - 使用哈希表优化标签合并
    // "FEC1_1"           - FEC1_1 版本
    // "FECunion"         - 新导入的 FECunion 版本
    // "FEC_Union"        - 使用并查集替代标签/哈希表合并
    // "EC"               - PCL 原生 EuclideanClusterExtraction
    // "RG"               - Region Growing
    // "FEC_Block"        - FEC 的分块并行版本
    // "FEC1_Block"       - FEC1 的分块并行版本
    // "FEC_Union_Block"  - FEC_Union 的分块并行版本
    // "FEC_Union_Block_new" - 新版 FEC_Union_Block
    // "FEC_Union_Block_new2" - 在 new 基础上继续优化 final 聚合
    // "FECunion_Block"  - 新导入的 FECunion_Block 版本
    // "FEC_Union_Grid_Block" - FEC_Union 的网格哈希分块并行版本
    // "Voxel_FEC1"       - 基于体素哈希搜索的 FEC1 版本
    // "EC_Block"         - EC 的分块并行版本
    // 单算法模式下，直接修改这两个字符串即可：
    // single_dataset_path 选择数据集
    // single_algorithm    选择算法
    const std::string single_algorithm = "FEC_Union_Block_new2";
    const std::string single_dataset_path = "./data/046.ply";
    // 通用参数
    int min_cluster_size = 100;
    double tolerance = 0.2; 
    int max_n = 50;
    const std::vector<double> experiment_tolerances = {0.1, 0.2, 0.3, 0.4, 0.5};

    // RG (区域生长) 专属参数
    float smoothness_threshold = 3.0; // 角度
    float curvature_threshold = 1.0;  // 曲率
    // --------------------------------------------------

    const std::vector<std::string> algorithms = {
        "FEC",
        "FEC1",
        "FEC1_1",
        "FECunion",
        "FEC_Union",
        "EC",
        "RG",
        "FEC_Block",
        "FEC1_Block",
        "FEC_Union_Block",
        "FEC_Union_Block_new",
        "FEC_Union_Block_new2",
        "FECunion_Block",
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

    std::string runtime_dataset_path = single_dataset_path;
    std::vector<std::string> runtime_algorithms = {single_algorithm};
    std::vector<double> runtime_tolerances = {tolerance};
    bool runtime_visualize = enable_visualization;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            runtime_dataset_path = argv[++i];
        } else if (arg == "--alg" && i + 1 < argc) {
            runtime_algorithms = split_csv(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            runtime_tolerances.clear();
            for (const auto& token : split_csv(argv[++i])) {
                try {
                    runtime_tolerances.push_back(std::stod(token));
                } catch (...) {
                    cout << "Invalid tolerance: " << token << endl;
                    return -1;
                }
            }
            if (runtime_tolerances.empty()) {
                runtime_tolerances.push_back(tolerance);
            }
        } else if (arg == "--visualize" && i + 1 < argc) {
            runtime_visualize = parse_bool_arg(argv[++i], runtime_visualize);
        } else if (arg == "--help") {
            cout << "Usage:\n"
                 << "  ./cpp/build/fec_run [--dataset PATH] [--alg NAME[,NAME...]] [--tol T[,T...]] [--visualize 0|1]\n\n"
                 << "Examples:\n"
                 << "  ./cpp/build/fec_run\n"
                 << "  ./cpp/build/fec_run --dataset ./data/046.ply --alg FEC_Union,FEC_Union_Block --tol 0.2\n"
                 << "  ./cpp/build/fec_run --dataset ./data/046.ply --alg FEC_Union_Block_new,FEC_Union_Block_new2 --tol 0.1,0.2,0.3\n";
            return 0;
        } else {
            cout << "Unknown argument: " << arg << endl;
            cout << "Use --help to see supported options." << endl;
            return -1;
        }
    }

    if (!std::filesystem::exists(runtime_dataset_path)) {
        std::filesystem::path fallback = std::filesystem::path("./data") / runtime_dataset_path;
        if (std::filesystem::exists(fallback)) {
            runtime_dataset_path = fallback.string();
        }
    }

    for (const auto& alg : runtime_algorithms) {
        if (std::find(algorithms.begin(), algorithms.end(), alg) == algorithms.end()) {
            cout << "Unknown algorithm: " << alg << endl;
            return -1;
        }
    }

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
        if (current_algorithm == "FECunion") {
            return FECunion(cloud, min_cluster_size, tolerance, max_n);
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
        if (current_algorithm == "FEC_Union_Block_new") {
            return FEC_Union_Block_new(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FEC_Union_Block_new2") {
            return FEC_Union_Block_new2(cloud, min_cluster_size, tolerance, max_n);
        }
        if (current_algorithm == "FECunion_Block") {
            return FECunion_Block(cloud, min_cluster_size, tolerance, max_n);
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

    auto measure_algorithm = [&](pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                 const std::string& current_algorithm,
                                 bool verbose) -> std::pair<RunMetrics, std::vector<pcl::PointIndices>> {
        using Clock = std::chrono::steady_clock;

        std::vector<pcl::PointIndices> cluster_indices;
        std::vector<RunMetrics> runs;
        runs.reserve(repeat_runs);

        if (verbose) {
            cout << "Algorithm: " << current_algorithm << endl;
            cout << "Repeat runs: " << repeat_runs << " (warmup: " << warmup_runs << ")" << endl;
            cout << "Average rule: drop warmup, then drop max/min by total" << endl;
        }

        for (int run = 0; run < repeat_runs; ++run) {
            std::ostringstream captured;
            std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());
            auto t0 = Clock::now();
            cluster_indices = run_algorithm(cloud, current_algorithm);
            auto t1 = Clock::now();
            std::cout.rdbuf(old_buf);

            const std::string algorithm_output = captured.str();
            if (verbose && !algorithm_output.empty()) {
                cout << algorithm_output;
                if (algorithm_output.back() != '\n') {
                    cout << '\n';
                }
            }

            RunMetrics metrics;
            metrics.total_ms = extract_metric(algorithm_output, "total").value_or(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
            metrics.build_ms = extract_metric(algorithm_output, "build").value_or(0.0);
            metrics.search_ms = extract_metric(algorithm_output, "search").value_or(0.0);
            metrics.merge_ms = extract_metric(algorithm_output, "merge").value_or(0.0);
            metrics.final_ms = extract_metric(algorithm_output, "final").value_or(0.0);
            metrics.clusters = extract_cluster_count(algorithm_output).value_or(static_cast<int>(cluster_indices.size()));
            runs.push_back(metrics);

            if (verbose) {
                cout << "Run " << (run + 1)
                     << " total= " << metrics.total_ms << " ms"
                     << " clusters= " << metrics.clusters;
                if (run < warmup_runs) {
                    cout << " [warmup]";
                }
                cout << endl;
            }
        }

        int start_index = std::min(warmup_runs, repeat_runs);
        std::vector<int> kept_indices;
        kept_indices.reserve(repeat_runs - start_index);
        for (int i = start_index; i < repeat_runs; ++i) {
            kept_indices.push_back(i);
        }

        int min_index = -1;
        int max_index = -1;
        if (kept_indices.size() > 2) {
            auto min_it = std::min_element(
                kept_indices.begin(), kept_indices.end(),
                [&](int a, int b) { return runs[a].total_ms < runs[b].total_ms; });
            min_index = *min_it;
            kept_indices.erase(min_it);

            auto max_it = std::max_element(
                kept_indices.begin(), kept_indices.end(),
                [&](int a, int b) { return runs[a].total_ms < runs[b].total_ms; });
            max_index = *max_it;
            kept_indices.erase(max_it);
        }

        if (verbose && min_index >= 0 && max_index >= 0) {
            cout << "Dropped runs after warmup: min=Run " << (min_index + 1)
                 << " max=Run " << (max_index + 1) << endl;
        }

        RunMetrics avg;
        for (int idx : kept_indices) {
            avg.total_ms += runs[idx].total_ms;
            avg.build_ms += runs[idx].build_ms;
            avg.search_ms += runs[idx].search_ms;
            avg.merge_ms += runs[idx].merge_ms;
            avg.final_ms += runs[idx].final_ms;
            avg.clusters += runs[idx].clusters;
        }

        if (!kept_indices.empty()) {
            const double denom = static_cast<double>(kept_indices.size());
            avg.total_ms /= denom;
            avg.build_ms /= denom;
            avg.search_ms /= denom;
            avg.merge_ms /= denom;
            avg.final_ms /= denom;
            avg.clusters = static_cast<int>(std::lround(static_cast<double>(avg.clusters) / denom));
        }

        return {avg, cluster_indices};
    };

    if (run_all_datasets) {
        for (const auto& dataset_path : datasets) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_cloud(dataset_path);
            if (!cloud) {
                cout << "Dataset: " << dataset_path << " load failed." << endl;
                continue;
            }

            const std::string dataset_name = std::filesystem::path(dataset_path).filename().string();
            for (double current_tolerance : experiment_tolerances) {
                tolerance = current_tolerance;
                cout << "\nDataset: " << dataset_name
                     << " Points: " << cloud->size()
                     << " Tolerance: " << std::fixed << std::setprecision(1) << current_tolerance << endl;
                cout << "Repeat runs: " << repeat_runs << " (warmup: " << warmup_runs << ")" << endl;
                cout << "Average rule: drop warmup, then drop max/min by total" << endl;

                RunMetrics fastest_metrics;
                std::string fastest_algorithm;
                bool fastest_set = false;
                for (const auto& algorithm : algorithms) {
                    auto [avg, ignored_clusters] = measure_algorithm(cloud, algorithm, false);
                    cout << std::fixed << std::setprecision(3)
                         << algorithm
                         << " total= " << avg.total_ms << " ms "
                         << "build= " << avg.build_ms << " ms "
                         << "search= " << avg.search_ms << " ms "
                         << "merge= " << avg.merge_ms << " ms "
                         << "final= " << avg.final_ms << " ms "
                         << "clusters=" << avg.clusters << "\n";
                    if (!fastest_set || avg.total_ms < fastest_metrics.total_ms) {
                        fastest_set = true;
                        fastest_metrics = avg;
                        fastest_algorithm = algorithm;
                    }
                }
                if (fastest_set) {
                    cout << "Fastest: " << fastest_algorithm
                         << " total= " << fastest_metrics.total_ms << " ms"
                         << " clusters=" << fastest_metrics.clusters << "\n";
                }
            }
        }
        return 0;
    }

    // 1. 读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_cloud(runtime_dataset_path);
    if (!cloud) {
        return -1;
    }
    const std::string dataset_name = std::filesystem::path(runtime_dataset_path).filename().string();
    cout << "Dataset: " << dataset_name << endl;
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

    if (runtime_algorithms.size() == 1 && runtime_tolerances.size() == 1) {
        tolerance = runtime_tolerances.front();
        cout << "Dataset: " << dataset_name << endl;
        cout << "Tolerance: " << std::fixed << std::setprecision(3) << tolerance << endl;
        auto [avg, cluster_indices] = measure_algorithm(cloud, runtime_algorithms.front(), true);

        cout << std::fixed << std::setprecision(3)
             << "Average (trimmed, excluding warmup): total= " << avg.total_ms << " ms "
             << "build= " << avg.build_ms << " ms "
             << "search= " << avg.search_ms << " ms "
             << "merge= " << avg.merge_ms << " ms "
             << "final= " << avg.final_ms << " ms "
             << "clusters= " << avg.clusters << endl;

        if (runtime_visualize) {
            visualize_clusters(cloud, cluster_indices, "Algorithm Test: " + runtime_algorithms.front());
        }
        return 0;
    }

    cout << "Repeat runs: " << repeat_runs << " (warmup: " << warmup_runs << ")" << endl;
    cout << "Average rule: drop warmup, then drop max/min by total" << endl;

    for (double current_tolerance : runtime_tolerances) {
        tolerance = current_tolerance;
        cout << "\nTolerance: " << std::fixed << std::setprecision(3) << current_tolerance << endl;
        for (const auto& algorithm : runtime_algorithms) {
            auto [avg, ignored_clusters] = measure_algorithm(cloud, algorithm, false);
            cout << std::fixed << std::setprecision(3)
                 << algorithm
                 << " total= " << avg.total_ms << " ms "
                 << "build= " << avg.build_ms << " ms "
                 << "search= " << avg.search_ms << " ms "
                 << "merge= " << avg.merge_ms << " ms "
                 << "final= " << avg.final_ms << " ms "
                 << "clusters=" << avg.clusters << "\n";
        }
    }

    return 0;
}
