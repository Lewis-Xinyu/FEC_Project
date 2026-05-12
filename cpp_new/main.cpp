#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/PointIndices.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "FECs.h"
#include "FEC1.h"
#include "FECunion.h"

namespace fs = std::filesystem;

// ======================== 参数区：按当前实验需求改这里 ========================
static const std::string INPUT_DIR  = "reports/semantic_kitti_seq00_static";
static const std::string OUTPUT_DIR = "reports/cpp_new_multiframe_timing";

static constexpr int MIN_COMPONENT_SIZE = 100;
static constexpr double TOLORANCE       = 0.2;
static const std::vector<int> MAX_N_VALUES = {0, 50};

// 正式测速：总共 7 次，去掉最小/最大后统计中间 5 次。
static constexpr int WARMUP_RUNS = 0;
static constexpr int FORMAL_RUNS = 7;   // 去掉最大最小后 kept=5
// ============================================================================

struct RunStats {
    double total_ms  = 0.0;
    double build_ms  = 0.0;
    double search_ms = 0.0;
    double merge_ms  = 0.0;
    double final_ms  = 0.0;
    int clusters     = 0;
};

struct SummaryStats {
    double mean = 0.0;
    double stddev = 0.0;
    double minv = 0.0;
    double maxv = 0.0;
};

struct AlgoSummary {
    std::string file;
    std::string algorithm;
    int points = 0;
    int kept_runs = 0;
    int clusters_median = 0;
    SummaryStats total;
    SummaryStats build;
    SummaryStats search;
    SummaryStats merge;
    SummaryStats final;
};

static std::string targetLabelFromPoints(int points) {
    if (points >= 1000000 && points % 1000000 == 0) {
        return std::to_string(points / 1000000) + "M";
    }
    return std::to_string(points / 1000) + "k";
}

static std::string makePlyName(int points) {
    return "seq00_static_" + targetLabelFromPoints(points) + ".ply";
}

static SummaryStats calcSummary(const std::vector<double>& values) {
    SummaryStats s;
    if (values.empty()) return s;

    s.minv = *std::min_element(values.begin(), values.end());
    s.maxv = *std::max_element(values.begin(), values.end());
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());

    double var = 0.0;
    for (double v : values) {
        const double d = v - s.mean;
        var += d * d;
    }
    if (values.size() > 1) {
        var /= static_cast<double>(values.size() - 1); // sample stddev
    }
    s.stddev = std::sqrt(var);
    return s;
}

static std::vector<int> keptIndicesDropMinMaxByTotal(const std::vector<RunStats>& runs) {
    std::vector<int> ids(runs.size());
    std::iota(ids.begin(), ids.end(), 0);

    if (runs.size() <= 2) return ids;

    auto min_it = std::min_element(ids.begin(), ids.end(), [&](int a, int b) {
        return runs[a].total_ms < runs[b].total_ms;
    });
    const int min_id = *min_it;

    auto max_it = std::max_element(ids.begin(), ids.end(), [&](int a, int b) {
        return runs[a].total_ms < runs[b].total_ms;
    });
    const int max_id = *max_it;

    std::vector<int> kept;
    kept.reserve(runs.size() - 2);
    for (int id : ids) {
        if (id != min_id && id != max_id) kept.push_back(id);
    }
    return kept;
}

static int medianClusterCount(const std::vector<RunStats>& runs, const std::vector<int>& kept) {
    std::vector<int> cs;
    cs.reserve(kept.size());
    for (int id : kept) cs.push_back(runs[id].clusters);
    if (cs.empty()) return 0;
    std::sort(cs.begin(), cs.end());
    return cs[cs.size() / 2];
}

static AlgoSummary summarizeRuns(
    const std::string& file,
    const std::string& algorithm,
    int points,
    const std::vector<RunStats>& runs
) {
    const std::vector<int> kept = keptIndicesDropMinMaxByTotal(runs);

    std::vector<double> total, build, search, merge, final;
    total.reserve(kept.size());
    build.reserve(kept.size());
    search.reserve(kept.size());
    merge.reserve(kept.size());
    final.reserve(kept.size());

    for (int id : kept) {
        total.push_back(runs[id].total_ms);
        build.push_back(runs[id].build_ms);
        search.push_back(runs[id].search_ms);
        merge.push_back(runs[id].merge_ms);
        final.push_back(runs[id].final_ms);
    }

    AlgoSummary s;
    s.file = file;
    s.algorithm = algorithm;
    s.points = points;
    s.kept_runs = static_cast<int>(kept.size());
    s.clusters_median = medianClusterCount(runs, kept);
    s.total = calcSummary(total);
    s.build = calcSummary(build);
    s.search = calcSummary(search);
    s.merge = calcSummary(merge);
    s.final = calcSummary(final);
    return s;
}

static void writeSummaryHeader(std::ofstream& out) {
    out << "file,target_points,algorithm,max_n,points,kept_runs,clusters_median,"
        << "total_mean_ms,total_std_ms,total_min_ms,total_max_ms,"
        << "build_mean_ms,build_std_ms,build_min_ms,build_max_ms,"
        << "search_mean_ms,search_std_ms,search_min_ms,search_max_ms,"
        << "merge_mean_ms,merge_std_ms,merge_min_ms,merge_max_ms,"
        << "final_mean_ms,final_std_ms,final_min_ms,final_max_ms\n";
}

static void writeSummaryRow(std::ofstream& out, const AlgoSummary& s, int target_points, int max_n) {
    out << s.file << ',' << target_points << ',' << s.algorithm << ',' << max_n << ',' << s.points << ',' << s.kept_runs << ',' << s.clusters_median << ','
        << s.total.mean  << ',' << s.total.stddev  << ',' << s.total.minv  << ',' << s.total.maxv  << ','
        << s.build.mean  << ',' << s.build.stddev  << ',' << s.build.minv  << ',' << s.build.maxv  << ','
        << s.search.mean << ',' << s.search.stddev << ',' << s.search.minv << ',' << s.search.maxv << ','
        << s.merge.mean  << ',' << s.merge.stddev  << ',' << s.merge.minv  << ',' << s.merge.maxv  << ','
        << s.final.mean  << ',' << s.final.stddev  << ',' << s.final.minv  << ',' << s.final.maxv  << '\n';
}

static void writeRawHeader(std::ofstream& out) {
    out << "file,target_points,algorithm,max_n,points,run_id,total_ms,build_ms,search_ms,merge_ms,final_ms,clusters\n";
}

static void writeRawRow(
    std::ofstream& out,
    const std::string& file,
    int target_points,
    const std::string& algorithm,
    int max_n,
    int points,
    int run_id,
    const RunStats& s
) {
    out << file << ',' << target_points << ',' << algorithm << ',' << max_n << ',' << points << ',' << run_id << ','
        << s.total_ms << ',' << s.build_ms << ',' << s.search_ms << ','
        << s.merge_ms << ',' << s.final_ms << ',' << s.clusters << '\n';
}

static void printSummaryLine(const AlgoSummary& s, int max_n) {
    std::cout << std::fixed << std::setprecision(3)
              << "  " << std::setw(9) << s.algorithm
              << " | max_n=" << std::setw(3) << max_n
              << " | clusters=" << std::setw(5) << s.clusters_median
              << " | total=" << s.total.mean << " ± " << s.total.stddev
              << " ms [" << s.total.minv << ", " << s.total.maxv << "]"
              << " | build=" << s.build.mean << " ± " << s.build.stddev
              << " | search=" << s.search.mean << " ± " << s.search.stddev
              << " | merge=" << s.merge.mean << " ± " << s.merge.stddev
              << " | final=" << s.final.mean << " ± " << s.final.stddev
              << '\n';
}

int main() {
    fs::create_directories(OUTPUT_DIR);

    const fs::path summary_path = fs::path(OUTPUT_DIR) / "timing_summary.csv";
    const fs::path raw_path     = fs::path(OUTPUT_DIR) / "timing_raw_runs.csv";

    std::ofstream summary_csv(summary_path.string());
    std::ofstream raw_csv(raw_path.string());
    if (!summary_csv.is_open() || !raw_csv.is_open()) {
        std::cerr << "[Error] Cannot open output CSV files in: " << OUTPUT_DIR << '\n';
        return 1;
    }

    summary_csv << std::fixed << std::setprecision(6);
    raw_csv << std::fixed << std::setprecision(6);
    writeSummaryHeader(summary_csv);
    writeRawHeader(raw_csv);

    std::cout << "============================================================\n";
    std::cout << " Scientific timing test: FEC / FEC1 / FECunion\n";
    std::cout << " Input : " << INPUT_DIR << "/seq00_static_50k.ply ... seq00_static_3M.ply\n";
    std::cout << " Output: " << OUTPUT_DIR << "\n";
    std::cout << " Params: min_size=" << MIN_COMPONENT_SIZE
              << ", tolerance=" << TOLORANCE
              << ", max_n={0,50}\n";
    std::cout << " Runs  : warmup=" << WARMUP_RUNS
              << ", formal=" << FORMAL_RUNS
              << ", kept=" << (FORMAL_RUNS > 2 ? FORMAL_RUNS - 2 : FORMAL_RUNS)
              << " after dropping min/max by total time\n";
    std::cout << "============================================================\n";

    for (int target_points = 50000; target_points <= 3000000; target_points += 50000) {
        const std::string file_name = makePlyName(target_points);
        const fs::path ply_path = fs::path(INPUT_DIR) / file_name;

        if (!fs::exists(ply_path)) {
            std::cout << "[Skip] " << file_name << " not found.\n";
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(ply_path.string(), *cloud) != 0) {
            std::cout << "[Skip] failed to load " << ply_path.string() << '\n';
            continue;
        }

        const int points = static_cast<int>(cloud->size());
        std::cout << "\nFile " << file_name << " | points=" << points << '\n';

        for (int max_n : MAX_N_VALUES) {
            // ------------------------------ FEC ------------------------------
            for (int w = 0; w < WARMUP_RUNS; ++w) {
                FECStageStats tmp;
                volatile std::size_t sink = FEC(cloud, MIN_COMPONENT_SIZE, TOLORANCE, max_n, &tmp).size();
                (void)sink;
            }
            std::vector<RunStats> fec_runs;
            fec_runs.reserve(FORMAL_RUNS);
            for (int r = 0; r < FORMAL_RUNS; ++r) {
                FECStageStats st;
                auto clusters = FEC(cloud, MIN_COMPONENT_SIZE, TOLORANCE, max_n, &st);
                RunStats rs{st.total_ms, st.build_ms, st.search_ms, st.merge_ms, st.final_ms, static_cast<int>(clusters.size())};
                fec_runs.push_back(rs);
                writeRawRow(raw_csv, file_name, target_points, "FEC", max_n, points, r + 1, rs);
            }
            AlgoSummary fec_sum = summarizeRuns(file_name, "FEC", points, fec_runs);
            writeSummaryRow(summary_csv, fec_sum, target_points, max_n);
            printSummaryLine(fec_sum, max_n);

            // ------------------------------ FEC1 -----------------------------
            for (int w = 0; w < WARMUP_RUNS; ++w) {
                FEC1StageStats tmp;
                volatile std::size_t sink = FEC1(cloud, MIN_COMPONENT_SIZE, TOLORANCE, max_n, &tmp).size();
                (void)sink;
            }
            std::vector<RunStats> fec1_runs;
            fec1_runs.reserve(FORMAL_RUNS);
            for (int r = 0; r < FORMAL_RUNS; ++r) {
                FEC1StageStats st;
                auto clusters = FEC1(cloud, MIN_COMPONENT_SIZE, TOLORANCE, max_n, &st);
                RunStats rs{st.total_ms, st.build_ms, st.search_ms, st.merge_ms, st.final_ms, static_cast<int>(clusters.size())};
                fec1_runs.push_back(rs);
                writeRawRow(raw_csv, file_name, target_points, "FEC1", max_n, points, r + 1, rs);
            }
            AlgoSummary fec1_sum = summarizeRuns(file_name, "FEC1", points, fec1_runs);
            writeSummaryRow(summary_csv, fec1_sum, target_points, max_n);
            printSummaryLine(fec1_sum, max_n);

            // ---------------------------- FECunion ---------------------------
            for (int w = 0; w < WARMUP_RUNS; ++w) {
                FECunionStageStats tmp;
                volatile std::size_t sink = FECunion(cloud, MIN_COMPONENT_SIZE, TOLORANCE, max_n, &tmp).size();
                (void)sink;
            }
            std::vector<RunStats> fecunion_runs;
            fecunion_runs.reserve(FORMAL_RUNS);
            for (int r = 0; r < FORMAL_RUNS; ++r) {
                FECunionStageStats st;
                auto clusters = FECunion(cloud, MIN_COMPONENT_SIZE, TOLORANCE, max_n, &st);
                RunStats rs{st.total_ms, st.build_ms, st.search_ms, st.merge_ms, st.final_ms, static_cast<int>(clusters.size())};
                fecunion_runs.push_back(rs);
                writeRawRow(raw_csv, file_name, target_points, "FECunion", max_n, points, r + 1, rs);
            }
            AlgoSummary fecunion_sum = summarizeRuns(file_name, "FECunion", points, fecunion_runs);
            writeSummaryRow(summary_csv, fecunion_sum, target_points, max_n);
            printSummaryLine(fecunion_sum, max_n);
        }
    }

    std::cout << "\n[Done] summary CSV: " << summary_path.string() << '\n';
    std::cout << "[Done] raw CSV    : " << raw_path.string() << '\n';
    return 0;
}
