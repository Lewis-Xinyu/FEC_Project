#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <chrono>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "FEC_Union.h"
#include "FEC_Union_Block.h"

// ========================
// 配置区：按需修改
// ========================
static const std::string kPlyPath = "./data/046.ply";
static const int kMinComponentSize = 100;
static const int kMaxN = 50;

// 你想测试的 tolerance
static const std::vector<double> kTolerances = {0.1, 0.2, 0.3, 0.4, 0.5};

// 每个算法重复运行次数
static const int kRepeat = 7;

// ========================
// 工具函数
// ========================
double trimmed_mean_7(std::vector<double> v) {
    if (v.size() != 7) return 0.0;
    std::sort(v.begin(), v.end());
    double sum = 0.0;
    for (int i = 1; i <= 5; ++i) sum += v[i];
    return sum / 5.0;
}

int median_of_7(std::vector<int> v) {
    if (v.size() != 7) return 0;
    std::sort(v.begin(), v.end());
    return v[3];
}

// 屏蔽算法内部 cout 输出，同时统计算法总耗时
template <typename Func>
double run_and_measure_ms(Func&& func, int& cluster_count) {
    using Clock = std::chrono::high_resolution_clock;

    std::streambuf* old_buf = std::cout.rdbuf();
    std::ostringstream temp_out;
    std::cout.rdbuf(temp_out.rdbuf());

    auto t0 = Clock::now();
    auto clusters = func();
    auto t1 = Clock::now();

    std::cout.rdbuf(old_buf);

    cluster_count = static_cast<int>(clusters.size());
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct RowResult {
    double tolerance = 0.0;

    double fec_union_avg_ms = 0.0;
    int    fec_union_clusters = 0;

    double fec_union_block_avg_ms = 0.0;
    int    fec_union_block_clusters = 0;

    double avg_difference = 0.0;      // block - union
    int    cluster_difference = 0;    // block - union
    double speedup = 0.0;             // union / block
    std::string winner;               // Faster one
};

std::string repeat_char(char ch, int n) {
    return std::string(n, ch);
}

void print_title(const std::string& text, int width = 110) {
    int pad = std::max(0, width - static_cast<int>(text.size()) - 2);
    int left = pad / 2;
    int right = pad - left;
    std::cout << "\n" << repeat_char('=', width) << "\n";
    std::cout << repeat_char(' ', left) << text << repeat_char(' ', right) << "\n";
    std::cout << repeat_char('=', width) << "\n";
}

void print_kv(const std::string& key, const std::string& value, int keyw = 22) {
    std::cout << "  " << std::left << std::setw(keyw) << key << ": " << value << "\n";
}

int main() {
    using CloudT = pcl::PointCloud<pcl::PointXYZ>;
    CloudT::Ptr cloud(new CloudT);

    if (pcl::io::loadPLYFile<pcl::PointXYZ>(kPlyPath, *cloud) != 0) {
        std::cerr << "Failed to load PLY file: " << kPlyPath << std::endl;
        return -1;
    }

    print_title("FEC_Union vs FEC_Union_Block Tolerance Benchmark");

    print_kv("PLY file", kPlyPath);
    print_kv("Points", std::to_string(cloud->size()));
    print_kv("Min component size", std::to_string(kMinComponentSize));
    print_kv("Max neighbors (max_n)", std::to_string(kMaxN));
    print_kv("Repeat times", std::to_string(kRepeat));

    {
        std::ostringstream oss;
        for (std::size_t i = 0; i < kTolerances.size(); ++i) {
            if (i) oss << ", ";
            oss << std::fixed << std::setprecision(1) << kTolerances[i];
        }
        print_kv("Tolerances", oss.str());
    }

#ifdef _OPENMP
    print_kv("_OPENMP", std::to_string(_OPENMP));
#endif

    std::vector<RowResult> results;
    results.reserve(kTolerances.size());

    for (double tol : kTolerances) {
        std::vector<double> union_times;
        std::vector<int> union_clusters;
        std::vector<double> block_times;
        std::vector<int> block_clusters;

        union_times.reserve(kRepeat);
        union_clusters.reserve(kRepeat);
        block_times.reserve(kRepeat);
        block_clusters.reserve(kRepeat);

        for (int i = 0; i < kRepeat; ++i) {
            int c1 = 0;
            double t1 = run_and_measure_ms([&]() {
                return FEC_Union(cloud, kMinComponentSize, tol, kMaxN);
            }, c1);

            union_times.push_back(t1);
            union_clusters.push_back(c1);

            int c2 = 0;
            double t2 = run_and_measure_ms([&]() {
                return FEC_Union_Block(cloud, kMinComponentSize, tol, kMaxN);
            }, c2);

            block_times.push_back(t2);
            block_clusters.push_back(c2);
        }

        RowResult row;
        row.tolerance = tol;
        row.fec_union_avg_ms = trimmed_mean_7(union_times);
        row.fec_union_clusters = median_of_7(union_clusters);

        row.fec_union_block_avg_ms = trimmed_mean_7(block_times);
        row.fec_union_block_clusters = median_of_7(block_clusters);

        row.avg_difference = row.fec_union_block_avg_ms - row.fec_union_avg_ms;
        row.cluster_difference = row.fec_union_block_clusters - row.fec_union_clusters;

        if (row.fec_union_block_avg_ms > 1e-12) {
            row.speedup = row.fec_union_avg_ms / row.fec_union_block_avg_ms;
        } else {
            row.speedup = 0.0;
        }

        if (row.fec_union_block_avg_ms < row.fec_union_avg_ms) {
            row.winner = "Block";
        } else if (row.fec_union_block_avg_ms > row.fec_union_avg_ms) {
            row.winner = "Union";
        } else {
            row.winner = "Tie";
        }

        results.push_back(row);
    }

    // ========================
    // 美观表格输出
    // ========================
    print_title("Benchmark Result Table");

    const int w_tol      = 10;
    const int w_u_ms     = 16;
    const int w_u_cl     = 14;
    const int w_b_ms     = 16;
    const int w_b_cl     = 14;
    const int w_diff     = 14;
    const int w_cdiff    = 12;
    const int w_speedup  = 10;
    const int w_winner   = 10;

    auto sep = [&]() {
        std::cout
            << "+"
            << repeat_char('-', w_tol)     << "+"
            << repeat_char('-', w_u_ms)    << "+"
            << repeat_char('-', w_u_cl)    << "+"
            << repeat_char('-', w_b_ms)    << "+"
            << repeat_char('-', w_b_cl)    << "+"
            << repeat_char('-', w_diff)    << "+"
            << repeat_char('-', w_cdiff)   << "+"
            << repeat_char('-', w_speedup) << "+"
            << repeat_char('-', w_winner)  << "+"
            << "\n";
    };

    sep();
    std::cout
        << "|"
        << std::setw(w_tol)     << std::left  << " Tol"
        << "|"
        << std::setw(w_u_ms)    << std::left  << " Union(ms)"
        << "|"
        << std::setw(w_u_cl)    << std::left  << " Union Cls"
        << "|"
        << std::setw(w_b_ms)    << std::left  << " Block(ms)"
        << "|"
        << std::setw(w_b_cl)    << std::left  << " Block Cls"
        << "|"
        << std::setw(w_diff)    << std::left  << " Diff(ms)"
        << "|"
        << std::setw(w_cdiff)   << std::left  << " Diff Cls"
        << "|"
        << std::setw(w_speedup) << std::left  << " Speedup"
        << "|"
        << std::setw(w_winner)  << std::left  << " Faster"
        << "|"
        << "\n";
    sep();

    std::cout << std::fixed << std::setprecision(3);

    for (const auto& r : results) {
        std::cout
            << "|"
            << std::setw(w_tol)     << std::right << r.tolerance
            << "|"
            << std::setw(w_u_ms)    << std::right << r.fec_union_avg_ms
            << "|"
            << std::setw(w_u_cl)    << std::right << r.fec_union_clusters
            << "|"
            << std::setw(w_b_ms)    << std::right << r.fec_union_block_avg_ms
            << "|"
            << std::setw(w_b_cl)    << std::right << r.fec_union_block_clusters
            << "|"
            << std::setw(w_diff)    << std::right << r.avg_difference
            << "|"
            << std::setw(w_cdiff)   << std::right << r.cluster_difference
            << "|"
            << std::setw(w_speedup) << std::right << r.speedup
            << "|"
            << std::setw(w_winner)  << std::right << r.winner
            << "|"
            << "\n";
    }
    sep();

    // ========================
    // 总结输出
    // ========================
    int block_win = 0;
    int union_win = 0;
    int tie_count = 0;

    double best_tol = 0.0;
    double best_speedup = 0.0;

    for (const auto& r : results) {
        if (r.winner == "Block") {
            ++block_win;
            if (r.speedup > best_speedup) {
                best_speedup = r.speedup;
                best_tol = r.tolerance;
            }
        } else if (r.winner == "Union") {
            ++union_win;
        } else {
            ++tie_count;
        }
    }

    print_title("Summary");
    print_kv("Block faster count", std::to_string(block_win));
    print_kv("Union faster count", std::to_string(union_win));
    print_kv("Tie count", std::to_string(tie_count));

    if (block_win > 0) {
        std::ostringstream oss;
        oss << "tol=" << std::fixed << std::setprecision(1) << best_tol
            << ", speedup=" << std::setprecision(3) << best_speedup << "x";
        print_kv("Best block result", oss.str());
    } else {
        print_kv("Best block result", "No tolerance beat Union");
    }

    std::cout << "\n说明：\n";
    std::cout << "  1) 每个 tolerance 下，每个算法运行 7 次。\n";
    std::cout << "  2) Avg(ms) = 去掉最大值和最小值后，对剩余 5 次求平均。\n";
    std::cout << "  3) Clusters = 7 次结果的中位数。\n";
    std::cout << "  4) Diff(ms) = Block(ms) - Union(ms)，负数表示 Block 更快。\n";
    std::cout << "  5) Speedup = Union(ms) / Block(ms)，大于 1 表示 Block 更快。\n\n";

    return 0;
}
