#pragma warning(disable:4996)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

using std::cerr;
using std::cout;
using std::endl;

struct Args {
    std::string root = "/mnt/d/semanticposs";
    std::string sequence = "00";
    std::string frame = "000000";
    std::string algorithm = "FEC_Union_Block_new2";
    std::string report_dir;
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int min_gt_points = 30;
    bool evaluate_all = false;
    bool write_report = false;
    int limit = -1;
    int start_frame = -1;
    int end_frame = -1;
};

struct FrameEval {
    std::string frame_id;
    int total_points = 0;
    int gt_objects = 0;
    int predicted_clusters = 0;
    int matched_03 = 0;
    int matched_05 = 0;
    double mean_best_iou = 0.0;
    double mean_best_recall = 0.0;
    double mean_best_precision = 0.0;
    double clustering_ms = 0.0;
};

static std::string trim(const std::string& s) {
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

static bool parse_bool(const std::string& value, bool fallback) {
    std::string lower;
    lower.reserve(value.size());
    for (char ch : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
    return fallback;
}

static void print_help() {
    cout << "SemanticPOSS clustering evaluation\n"
         << "Usage:\n"
         << "  ./cpp/build/semanticposs_eval_run [options]\n\n"
         << "Options:\n"
         << "  --root PATH              SemanticPOSS root containing dataset/sequences/\n"
         << "  --sequence ID            Sequence id, e.g. 00\n"
         << "  --frame ID               Frame id, e.g. 000123\n"
         << "  --start-frame N          Inclusive start frame index\n"
         << "  --end-frame N            Inclusive end frame index\n"
         << "  --alg NAME               Algorithm name, same as fec_run\n"
         << "  --write-report 0|1       Write a markdown report file\n"
         << "  --report-dir PATH        Directory for markdown reports\n"
         << "  --tol VALUE              Distance tolerance\n"
         << "  --min-cluster-size N     Minimum predicted cluster size\n"
         << "  --max-n N                Max neighbors for radius search\n"
         << "  --min-gt-points N        Ignore GT instances smaller than this\n"
         << "  --all 0|1                Evaluate all frames under the sequence\n"
         << "  --limit N                Max number of frames when --all 1\n"
         << "  --help                   Show this help\n\n"
         << "Examples:\n"
         << "  ./cpp/build/semanticposs_eval_run --root /mnt/d/semanticposs --sequence 00 --frame 000046 --alg FEC_Union_Block_new2 --tol 0.2\n"
         << "  ./cpp/build/semanticposs_eval_run --root /mnt/d/semanticposs --sequence 00 --all 1 --limit 20 --alg FEC_Union_Block_new2 --tol 0.2\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            print_help();
            return false;
        }
        auto need_value = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << endl;
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (key == "--root") {
            auto value = need_value(key);
            if (!value) return false;
            args.root = *value;
        } else if (key == "--sequence") {
            auto value = need_value(key);
            if (!value) return false;
            args.sequence = *value;
        } else if (key == "--frame") {
            auto value = need_value(key);
            if (!value) return false;
            args.frame = *value;
        } else if (key == "--start-frame") {
            auto value = need_value(key);
            if (!value) return false;
            args.start_frame = std::stoi(*value);
        } else if (key == "--end-frame") {
            auto value = need_value(key);
            if (!value) return false;
            args.end_frame = std::stoi(*value);
        } else if (key == "--alg") {
            auto value = need_value(key);
            if (!value) return false;
            args.algorithm = *value;
        } else if (key == "--write-report") {
            auto value = need_value(key);
            if (!value) return false;
            args.write_report = parse_bool(*value, args.write_report);
        } else if (key == "--report-dir") {
            auto value = need_value(key);
            if (!value) return false;
            args.report_dir = trim(*value);
        } else if (key == "--tol") {
            auto value = need_value(key);
            if (!value) return false;
            args.tolerance = std::stod(*value);
        } else if (key == "--min-cluster-size") {
            auto value = need_value(key);
            if (!value) return false;
            args.min_cluster_size = std::stoi(*value);
        } else if (key == "--max-n") {
            auto value = need_value(key);
            if (!value) return false;
            args.max_n = std::stoi(*value);
        } else if (key == "--min-gt-points") {
            auto value = need_value(key);
            if (!value) return false;
            args.min_gt_points = std::stoi(*value);
        } else if (key == "--all") {
            auto value = need_value(key);
            if (!value) return false;
            args.evaluate_all = parse_bool(*value, args.evaluate_all);
        } else if (key == "--limit") {
            auto value = need_value(key);
            if (!value) return false;
            args.limit = std::stoi(*value);
        } else {
            cerr << "Unknown argument: " << key << endl;
            print_help();
            return false;
        }
    }
    return true;
}

static std::vector<pcl::PointIndices> run_algorithm(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const std::string& current_algorithm,
    int min_cluster_size,
    double tolerance,
    int max_n
) {
    if (current_algorithm == "FEC") return FEC(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Block") return FEC_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC1") return FEC1(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC1_1") return FEC1_1(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FECunion") return FECunion(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "EC") return EC(cloud, tolerance, min_cluster_size);
    if (current_algorithm == "EC_Block") return EC_Block(cloud, tolerance, min_cluster_size);
    if (current_algorithm == "RG") return RG(cloud, min_cluster_size, 30, 3.0f, 1.0f);
    if (current_algorithm == "FEC_Union") return FEC_Union(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Block") return FEC_Union_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Block_new") return FEC_Union_Block_new(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Block_new2") return FEC_Union_Block_new2(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FECunion_Block") return FECunion_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Grid_Block") return FEC_Union_Grid_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "Voxel_FEC1") return Voxel_FEC1(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC1_Block") return FEC1_Block(cloud, min_cluster_size, tolerance, max_n);

    throw std::runtime_error("Unknown algorithm: " + current_algorithm);
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_velodyne_bin_cloud(const std::filesystem::path& bin_path) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open point cloud: " + bin_path.string());
    }

    struct PointXYZI {
        float x, y, z, intensity;
    };

    in.seekg(0, std::ios::end);
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size % static_cast<std::streamsize>(sizeof(PointXYZI)) != 0) {
        throw std::runtime_error("Unexpected .bin size: " + bin_path.string());
    }

    const std::size_t point_count = static_cast<std::size_t>(file_size / sizeof(PointXYZI));
    std::vector<PointXYZI> raw(point_count);
    in.read(reinterpret_cast<char*>(raw.data()), file_size);

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(point_count);
    for (const auto& p : raw) {
        cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static std::vector<std::uint32_t> load_uint32_labels(const std::filesystem::path& label_path) {
    std::ifstream in(label_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open label file: " + label_path.string());
    }

    in.seekg(0, std::ios::end);
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
        throw std::runtime_error("Unexpected label file size: " + label_path.string());
    }

    std::vector<std::uint32_t> labels(static_cast<std::size_t>(file_size / sizeof(std::uint32_t)));
    in.read(reinterpret_cast<char*>(labels.data()), file_size);
    return labels;
}

static std::filesystem::path resolve_sequences_root(const std::filesystem::path& root) {
    if (std::filesystem::exists(root / "sequences")) {
        return root / "sequences";
    }
    if (std::filesystem::exists(root / "dataset" / "sequences")) {
        return root / "dataset" / "sequences";
    }
    throw std::runtime_error(
        "SemanticPOSS root is incomplete: " + root.string() +
        " (expected sequences/ or dataset/sequences/)"
    );
}

static std::vector<std::string> list_frame_ids(const std::filesystem::path& velodyne_dir) {
    if (!std::filesystem::exists(velodyne_dir)) {
        throw std::runtime_error("Velodyne directory not found: " + velodyne_dir.string());
    }

    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(velodyne_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".bin") continue;
        ids.push_back(entry.path().stem().string());
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static double fps_from_ms(double ms) {
    return ms > 1e-12 ? 1000.0 / ms : 0.0;
}

static std::string sanitize_filename(std::string text) {
    for (char& ch : text) {
        const bool keep = std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_';
        if (!keep) ch = '_';
    }
    return text;
}

static std::filesystem::path default_report_dir_from_argv0(const char* argv0) {
    std::filesystem::path exe_path = std::filesystem::weakly_canonical(std::filesystem::absolute(argv0));
    return exe_path.parent_path().parent_path().parent_path() / "reports";
}

static std::filesystem::path make_report_path(
    const Args& args,
    const std::vector<std::string>& frame_ids,
    const std::filesystem::path& report_dir
) {
    std::string frame_part;
    if (args.evaluate_all) {
        if (frame_ids.empty()) {
            frame_part = "all_empty";
        } else {
            frame_part = "all_" + frame_ids.front() + "_" + frame_ids.back();
        }
    } else {
        frame_part = args.frame;
    }
    const std::string file_name =
        "semanticposs_" + sanitize_filename(args.sequence) + "_" +
        sanitize_filename(args.algorithm) + "_" + sanitize_filename(frame_part) + ".md";
    return report_dir / file_name;
}

static void write_markdown_report(
    const Args& args,
    const std::vector<FrameEval>& results,
    const std::filesystem::path& report_path
) {
    std::ofstream out(report_path);
    if (!out) {
        throw std::runtime_error("Failed to write report: " + report_path.string());
    }

    long long total_points = 0;
    double avg_miou = 0.0;
    double avg_recall = 0.0;
    double avg_precision = 0.0;
    double avg_cluster_ms = 0.0;
    double avg_fps = 0.0;
    double min_fps = std::numeric_limits<double>::max();
    double max_fps = 0.0;
    int total_gt = 0;
    int total_pred = 0;
    int total_match_03 = 0;
    int total_match_05 = 0;

    for (const auto& r : results) {
        total_points += r.total_points;
        avg_miou += r.mean_best_iou;
        avg_recall += r.mean_best_recall;
        avg_precision += r.mean_best_precision;
        avg_cluster_ms += r.clustering_ms;
        const double fps = fps_from_ms(r.clustering_ms);
        avg_fps += fps;
        min_fps = std::min(min_fps, fps);
        max_fps = std::max(max_fps, fps);
        total_gt += r.gt_objects;
        total_pred += r.predicted_clusters;
        total_match_03 += r.matched_03;
        total_match_05 += r.matched_05;
    }

    const double denom = results.empty() ? 1.0 : static_cast<double>(results.size());
    avg_miou /= denom;
    avg_recall /= denom;
    avg_precision /= denom;
    avg_cluster_ms /= denom;
    avg_fps /= denom;
    if (results.empty()) {
        min_fps = 0.0;
    }

    out << "# SemanticPOSS Experiment Report\n\n";
    out << "## Setup\n\n";
    out << "- Root: `" << args.root << "`\n";
    out << "- Sequence: `" << args.sequence << "`\n";
    out << "- Algorithm: `" << args.algorithm << "`\n";
    out << "- Tolerance: `" << args.tolerance << "`\n";
    out << "- Min cluster size: `" << args.min_cluster_size << "`\n";
    out << "- Max neighbors: `" << args.max_n << "`\n";
    out << "- Min GT points: `" << args.min_gt_points << "`\n";
    out << "- Frames: `" << results.size() << "`\n";
    out << "- Total points: `" << total_points << "`\n";
    out << "- FPS definition: `FPS = 1000 / cluster_wall_ms`\n\n";

    out << "## Summary\n\n";
    out << "| Frames | Total Points | Total GT | Total Pred Clusters | avg_mIoU | avg_mRecall | avg_mPrecision | gt_match_rate@0.3 | gt_match_rate@0.5 | avg_cluster_wall (ms) | avg_frame_FPS | min_frame_FPS | max_frame_FPS |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    out << std::fixed << std::setprecision(3)
        << "|" << results.size()
        << "|" << total_points
        << "|" << total_gt
        << "|" << total_pred
        << "|" << avg_miou
        << "|" << avg_recall
        << "|" << avg_precision
        << "|" << (total_gt > 0 ? static_cast<double>(total_match_03) / total_gt : 0.0)
        << "|" << (total_gt > 0 ? static_cast<double>(total_match_05) / total_gt : 0.0)
        << "|" << avg_cluster_ms
        << "|" << avg_fps
        << "|" << min_fps
        << "|" << max_fps
        << "|\n\n";

    out << "## Per-frame Result\n\n";
    out << "| Frame | Points | GT | Pred Clusters | mIoU | mRecall | mPrecision | matched@0.3 | matched@0.5 | cluster_wall (ms) | FPS |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& r : results) {
        out << std::fixed << std::setprecision(3)
            << "|" << r.frame_id
            << "|" << r.total_points
            << "|" << r.gt_objects
            << "|" << r.predicted_clusters
            << "|" << r.mean_best_iou
            << "|" << r.mean_best_recall
            << "|" << r.mean_best_precision
            << "|" << r.matched_03
            << "|" << r.matched_05
            << "|" << r.clustering_ms
            << "|" << fps_from_ms(r.clustering_ms)
            << "|\n";
    }
}

static FrameEval evaluate_frame(const Args& args, const std::filesystem::path& sequence_root, const std::string& frame_id) {
    const std::filesystem::path bin_path = sequence_root / "velodyne" / (frame_id + ".bin");
    const std::filesystem::path label_path = sequence_root / "labels" / (frame_id + ".label");

    auto cloud = load_velodyne_bin_cloud(bin_path);
    auto packed_labels = load_uint32_labels(label_path);
    if (packed_labels.size() != cloud->size()) {
        throw std::runtime_error(
            "Point/label count mismatch in frame " + frame_id +
            ": points=" + std::to_string(cloud->size()) +
            " labels=" + std::to_string(packed_labels.size())
        );
    }

    std::unordered_map<std::uint32_t, int> gt_key_to_index;
    std::vector<std::vector<int>> gt_point_lists;
    std::vector<int> point_to_gt(cloud->size(), -1);

    for (int i = 0; i < static_cast<int>(packed_labels.size()); ++i) {
        const std::uint32_t packed = packed_labels[i];
        const std::uint16_t semantic_id = static_cast<std::uint16_t>(packed & 0xffffu);
        const std::uint16_t instance_id = static_cast<std::uint16_t>(packed >> 16);

        if (semantic_id == 0 || instance_id == 0) {
            continue;
        }

        const std::uint32_t gt_key = (static_cast<std::uint32_t>(instance_id) << 16) | semantic_id;
        auto it = gt_key_to_index.find(gt_key);
        if (it == gt_key_to_index.end()) {
            const int new_index = static_cast<int>(gt_point_lists.size());
            gt_key_to_index.emplace(gt_key, new_index);
            gt_point_lists.push_back({});
            it = gt_key_to_index.find(gt_key);
        }
        gt_point_lists[it->second].push_back(i);
    }

    std::vector<std::vector<int>> filtered_gt_point_lists;
    filtered_gt_point_lists.reserve(gt_point_lists.size());
    for (const auto& points : gt_point_lists) {
        if (static_cast<int>(points.size()) < args.min_gt_points) {
            continue;
        }
        const int gt_index = static_cast<int>(filtered_gt_point_lists.size());
        filtered_gt_point_lists.push_back(points);
        for (int point_idx : points) {
            point_to_gt[point_idx] = gt_index;
        }
    }

    auto cluster_begin = std::chrono::steady_clock::now();
    auto clusters = run_algorithm(cloud, args.algorithm, args.min_cluster_size, args.tolerance, args.max_n);
    auto cluster_end = std::chrono::steady_clock::now();
    const double clustering_ms = std::chrono::duration<double, std::milli>(cluster_end - cluster_begin).count();

    std::vector<double> best_iou(filtered_gt_point_lists.size(), 0.0);
    std::vector<double> best_recall(filtered_gt_point_lists.size(), 0.0);
    std::vector<double> best_precision(filtered_gt_point_lists.size(), 0.0);

    for (const auto& cluster : clusters) {
        if (cluster.indices.empty()) continue;

        std::unordered_map<int, int> overlap_count;
        for (int point_idx : cluster.indices) {
            if (point_idx < 0 || point_idx >= static_cast<int>(point_to_gt.size())) continue;
            const int gt_idx = point_to_gt[point_idx];
            if (gt_idx >= 0) {
                ++overlap_count[gt_idx];
            }
        }

        for (const auto& [gt_idx, inter] : overlap_count) {
            const int gt_size = static_cast<int>(filtered_gt_point_lists[gt_idx].size());
            const int pred_size = static_cast<int>(cluster.indices.size());
            const double union_size = static_cast<double>(gt_size + pred_size - inter);
            const double iou = union_size > 0.0 ? static_cast<double>(inter) / union_size : 0.0;
            if (iou > best_iou[gt_idx]) {
                best_iou[gt_idx] = iou;
                best_recall[gt_idx] = gt_size > 0 ? static_cast<double>(inter) / gt_size : 0.0;
                best_precision[gt_idx] = pred_size > 0 ? static_cast<double>(inter) / pred_size : 0.0;
            }
        }
    }

    FrameEval result;
    result.frame_id = frame_id;
    result.total_points = static_cast<int>(cloud->size());
    result.gt_objects = static_cast<int>(filtered_gt_point_lists.size());
    result.predicted_clusters = static_cast<int>(clusters.size());
    result.clustering_ms = clustering_ms;

    if (!filtered_gt_point_lists.empty()) {
        for (int i = 0; i < static_cast<int>(filtered_gt_point_lists.size()); ++i) {
            result.mean_best_iou += best_iou[i];
            result.mean_best_recall += best_recall[i];
            result.mean_best_precision += best_precision[i];
            if (best_iou[i] >= 0.3) ++result.matched_03;
            if (best_iou[i] >= 0.5) ++result.matched_05;
        }
        const double denom = static_cast<double>(filtered_gt_point_lists.size());
        result.mean_best_iou /= denom;
        result.mean_best_recall /= denom;
        result.mean_best_precision /= denom;
    }

    cout << std::fixed << std::setprecision(3)
         << "Eval frame=" << frame_id
         << " points=" << result.total_points
         << " gt=" << result.gt_objects
         << " pred_clusters=" << result.predicted_clusters
         << " mIoU=" << result.mean_best_iou
         << " mRecall=" << result.mean_best_recall
         << " mPrecision=" << result.mean_best_precision
         << " matched@0.3=" << result.matched_03
         << " matched@0.5=" << result.matched_05
         << " cluster_wall=" << result.clustering_ms << " ms"
         << " fps=" << fps_from_ms(result.clustering_ms)
         << "\n";

    return result;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 0;
    }

    try {
        const std::filesystem::path root(args.root);
        const std::filesystem::path sequences_root = resolve_sequences_root(root);
        const std::filesystem::path sequence_root = sequences_root / args.sequence;
        const std::filesystem::path velodyne_dir = sequence_root / "velodyne";
        const std::filesystem::path labels_dir = sequence_root / "labels";
        if (!std::filesystem::exists(velodyne_dir) || !std::filesystem::exists(labels_dir)) {
            cerr << "SemanticPOSS sequence is incomplete: " << sequence_root << endl;
            cerr << "Expected directories: velodyne/ and labels/" << endl;
            return 1;
        }

        cout << "SemanticPOSS Eval\n";
        cout << "Root: " << root.string() << "\n";
        cout << "Sequence: " << args.sequence << "\n";
        cout << "Algorithm: " << args.algorithm << "\n";
        cout << "Tolerance: " << args.tolerance << "\n";
        cout << "Min cluster size: " << args.min_cluster_size << "\n";
        cout << "Min GT points: " << args.min_gt_points << "\n";

        std::vector<std::string> frame_ids;
        if (args.evaluate_all) {
            frame_ids = list_frame_ids(velodyne_dir);
            if (args.start_frame >= 0 || args.end_frame >= 0) {
                const int start_frame = std::max(0, args.start_frame);
                const int end_frame = args.end_frame >= 0 ? args.end_frame : std::numeric_limits<int>::max();
                std::vector<std::string> filtered_frame_ids;
                filtered_frame_ids.reserve(frame_ids.size());
                for (const auto& frame_id : frame_ids) {
                    const int numeric_id = std::stoi(frame_id);
                    if (numeric_id >= start_frame && numeric_id <= end_frame) {
                        filtered_frame_ids.push_back(frame_id);
                    }
                }
                frame_ids = std::move(filtered_frame_ids);
            }
            if (args.limit > 0 && args.limit < static_cast<int>(frame_ids.size())) {
                frame_ids.resize(args.limit);
            }
        } else {
            frame_ids = {args.frame};
        }

        cout << "Frames to evaluate: " << frame_ids.size() << "\n\n";

        std::vector<FrameEval> results;
        results.reserve(frame_ids.size());
        for (const auto& frame_id : frame_ids) {
            results.push_back(evaluate_frame(args, sequence_root, frame_id));
        }

        if (results.empty()) {
            cout << "No frames evaluated.\n";
            return 0;
        }

        double avg_miou = 0.0;
        double avg_recall = 0.0;
        double avg_precision = 0.0;
        double avg_cluster_ms = 0.0;
        int total_gt = 0;
        int total_pred = 0;
        int total_match_03 = 0;
        int total_match_05 = 0;

        for (const auto& r : results) {
            avg_miou += r.mean_best_iou;
            avg_recall += r.mean_best_recall;
            avg_precision += r.mean_best_precision;
            avg_cluster_ms += r.clustering_ms;
            total_gt += r.gt_objects;
            total_pred += r.predicted_clusters;
            total_match_03 += r.matched_03;
            total_match_05 += r.matched_05;
        }

        const double denom = static_cast<double>(results.size());
        avg_miou /= denom;
        avg_recall /= denom;
        avg_precision /= denom;
        avg_cluster_ms /= denom;

        cout << "\nSummary\n";
        cout << std::fixed << std::setprecision(3)
             << "frames=" << results.size()
             << " total_points=" << [&]() {
                    long long total_points = 0;
                    for (const auto& r : results) total_points += r.total_points;
                    return total_points;
                }()
             << " total_gt=" << total_gt
             << " total_pred_clusters=" << total_pred
             << " avg_mIoU=" << avg_miou
             << " avg_mRecall=" << avg_recall
             << " avg_mPrecision=" << avg_precision
             << " gt_match_rate@0.3=" << (total_gt > 0 ? static_cast<double>(total_match_03) / total_gt : 0.0)
             << " gt_match_rate@0.5=" << (total_gt > 0 ? static_cast<double>(total_match_05) / total_gt : 0.0)
             << " avg_cluster_wall=" << avg_cluster_ms << " ms"
             << " avg_fps=" << fps_from_ms(avg_cluster_ms)
             << "\n";

        if (args.write_report) {
            const std::filesystem::path report_dir = args.report_dir.empty()
                ? default_report_dir_from_argv0(argv[0])
                : std::filesystem::path(args.report_dir);
            std::filesystem::create_directories(report_dir);
            const std::filesystem::path report_path = make_report_path(args, frame_ids, report_dir);
            write_markdown_report(args, results, report_path);
            cout << "Report written: " << report_path.string() << "\n";
        }
    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}
