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
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
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

namespace {

constexpr std::uint16_t kGroundSemanticId = 22;

struct Args {
    std::string root = "/mnt/d/semanticposs";
    std::string sequence = "00";
    std::vector<std::string> algorithms = {"FECunion", "EC", "RG", "FEC"};
    std::string report_dir;
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int min_gt_points = 30;
    int limit = -1;
    int start_frame = -1;
    int end_frame = -1;
    bool remove_ground = true;
    bool write_report = true;
};

struct FilteredFrame {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std::vector<std::uint32_t> labels;
    int raw_points = 0;
    int removed_ground = 0;
};

struct FrameMetrics {
    int raw_points = 0;
    int removed_ground = 0;
    int points = 0;
    int gt = 0;
    int pred = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double matched_iou_sum = 0.0;
    double best_iou_sum = 0.0;
    double wall_ms = 0.0;
};

struct Summary {
    int frames = 0;
    long long raw_points = 0;
    long long removed_ground = 0;
    long long points = 0;
    int gt = 0;
    int pred = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double matched_iou_sum = 0.0;
    double best_iou_sum = 0.0;
    double avg_time_ms = 0.0;
    double avg_frame_fps = 0.0;
};

static std::string trim(const std::string& s) {
    std::size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

static std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

static bool parse_bool(const std::string& value, bool fallback) {
    std::string lower;
    for (char ch : value) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
    return fallback;
}

static void print_help() {
    cout << "SemanticPOSS report-style clustering evaluation\n"
         << "Usage:\n"
         << "  ./cpp/build/semanticposs_report_eval_run [options]\n\n"
         << "Options:\n"
         << "  --root PATH              SemanticPOSS root containing dataset/sequences/\n"
         << "  --sequence ID            Sequence id, e.g. 00\n"
         << "  --alg NAME[,NAME...]     Algorithms to compare\n"
         << "  --tol VALUE              Distance tolerance\n"
         << "  --min-cluster-size N     Minimum predicted cluster size\n"
         << "  --max-n N                Max neighbors for radius search\n"
         << "  --min-gt-points N        Ignore GT instances smaller than this\n"
         << "  --remove-ground 0|1      Remove semantic id 22 before clustering\n"
         << "  --start-frame N          Inclusive start frame index\n"
         << "  --end-frame N            Inclusive end frame index\n"
         << "  --limit N                Max number of frames\n"
         << "  --write-report 0|1       Write markdown report\n"
         << "  --report-dir PATH        Directory for markdown reports\n";
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
            auto v = need_value(key); if (!v) return false; args.root = *v;
        } else if (key == "--sequence") {
            auto v = need_value(key); if (!v) return false; args.sequence = *v;
        } else if (key == "--alg") {
            auto v = need_value(key); if (!v) return false; args.algorithms = split_csv(*v);
        } else if (key == "--tol") {
            auto v = need_value(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need_value(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need_value(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else if (key == "--min-gt-points") {
            auto v = need_value(key); if (!v) return false; args.min_gt_points = std::stoi(*v);
        } else if (key == "--remove-ground") {
            auto v = need_value(key); if (!v) return false; args.remove_ground = parse_bool(*v, args.remove_ground);
        } else if (key == "--start-frame") {
            auto v = need_value(key); if (!v) return false; args.start_frame = std::stoi(*v);
        } else if (key == "--end-frame") {
            auto v = need_value(key); if (!v) return false; args.end_frame = std::stoi(*v);
        } else if (key == "--limit") {
            auto v = need_value(key); if (!v) return false; args.limit = std::stoi(*v);
        } else if (key == "--write-report") {
            auto v = need_value(key); if (!v) return false; args.write_report = parse_bool(*v, args.write_report);
        } else if (key == "--report-dir") {
            auto v = need_value(key); if (!v) return false; args.report_dir = trim(*v);
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
    const std::string& algorithm,
    int min_cluster_size,
    double tolerance,
    int max_n
) {
    if (algorithm == "FEC") return FEC(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC_Block") return FEC_Block(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC1") return FEC1(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC1_1") return FEC1_1(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FECunion") return FECunion(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "EC") return EC(cloud, tolerance, min_cluster_size);
    if (algorithm == "EC_Block") return EC_Block(cloud, tolerance, min_cluster_size);
    if (algorithm == "RG") return RG(cloud, min_cluster_size, 30, 3.0f, 1.0f);
    if (algorithm == "FEC_Union") return FEC_Union(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC_Union_Block") return FEC_Union_Block(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC_Union_Block_new") return FEC_Union_Block_new(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC_Union_Block_new2") return FEC_Union_Block_new2(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FECunion_Block") return FECunion_Block(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC_Union_Grid_Block") return FEC_Union_Grid_Block(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "Voxel_FEC1") return Voxel_FEC1(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC1_Block") return FEC1_Block(cloud, min_cluster_size, tolerance, max_n);
    throw std::runtime_error("Unknown algorithm: " + algorithm);
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_velodyne_bin_cloud(const std::filesystem::path& bin_path) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open point cloud: " + bin_path.string());

    struct PointXYZI { float x, y, z, intensity; };
    in.seekg(0, std::ios::end);
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (file_size % static_cast<std::streamsize>(sizeof(PointXYZI)) != 0) {
        throw std::runtime_error("Unexpected .bin size: " + bin_path.string());
    }

    std::vector<PointXYZI> raw(static_cast<std::size_t>(file_size / sizeof(PointXYZI)));
    in.read(reinterpret_cast<char*>(raw.data()), file_size);

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(raw.size());
    for (const auto& p : raw) cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static std::vector<std::uint32_t> load_uint32_labels(const std::filesystem::path& label_path) {
    std::ifstream in(label_path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open label file: " + label_path.string());
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
    if (std::filesystem::exists(root / "sequences")) return root / "sequences";
    if (std::filesystem::exists(root / "dataset" / "sequences")) return root / "dataset" / "sequences";
    throw std::runtime_error("SemanticPOSS root is incomplete: " + root.string());
}

static std::vector<std::string> list_frame_ids(const std::filesystem::path& velodyne_dir, const Args& args) {
    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(velodyne_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".bin") continue;
        const std::string frame_id = entry.path().stem().string();
        const int numeric_id = std::stoi(frame_id);
        if (args.start_frame >= 0 && numeric_id < args.start_frame) continue;
        if (args.end_frame >= 0 && numeric_id > args.end_frame) continue;
        ids.push_back(frame_id);
    }
    std::sort(ids.begin(), ids.end());
    if (args.limit > 0 && args.limit < static_cast<int>(ids.size())) ids.resize(args.limit);
    return ids;
}

static double fps_from_ms(double ms) {
    return ms > 1e-12 ? 1000.0 / ms : 0.0;
}

static FilteredFrame load_frame(
    const std::filesystem::path& sequence_root,
    const std::string& frame_id,
    bool remove_ground
) {
    auto raw_cloud = load_velodyne_bin_cloud(sequence_root / "velodyne" / (frame_id + ".bin"));
    auto raw_labels = load_uint32_labels(sequence_root / "labels" / (frame_id + ".label"));
    if (raw_cloud->size() != raw_labels.size()) {
        throw std::runtime_error("Point/label count mismatch in frame " + frame_id);
    }

    FilteredFrame frame;
    frame.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    frame.cloud->reserve(raw_cloud->size());
    frame.labels.reserve(raw_labels.size());
    frame.raw_points = static_cast<int>(raw_cloud->size());

    for (int i = 0; i < static_cast<int>(raw_labels.size()); ++i) {
        const std::uint16_t semantic_id = static_cast<std::uint16_t>(raw_labels[i] & 0xffffu);
        if (remove_ground && semantic_id == kGroundSemanticId) {
            ++frame.removed_ground;
            continue;
        }
        frame.cloud->push_back(raw_cloud->points[i]);
        frame.labels.push_back(raw_labels[i]);
    }

    frame.cloud->width = static_cast<std::uint32_t>(frame.cloud->size());
    frame.cloud->height = 1;
    frame.cloud->is_dense = false;
    return frame;
}

static FrameMetrics evaluate_frame(
    const Args& args,
    const std::filesystem::path& sequence_root,
    const std::string& frame_id,
    const std::string& algorithm
) {
    FilteredFrame frame = load_frame(sequence_root, frame_id, args.remove_ground);

    std::unordered_map<std::uint32_t, int> gt_key_to_index;
    std::vector<std::vector<int>> gt_lists;
    std::vector<int> point_to_gt(frame.cloud->size(), -1);

    for (int i = 0; i < static_cast<int>(frame.labels.size()); ++i) {
        const std::uint32_t packed = frame.labels[i];
        const std::uint16_t semantic_id = static_cast<std::uint16_t>(packed & 0xffffu);
        const std::uint16_t instance_id = static_cast<std::uint16_t>(packed >> 16);
        if (semantic_id == 0 || instance_id == 0) continue;
        const std::uint32_t gt_key = (static_cast<std::uint32_t>(instance_id) << 16) | semantic_id;
        auto [it, inserted] = gt_key_to_index.emplace(gt_key, static_cast<int>(gt_lists.size()));
        if (inserted) gt_lists.push_back({});
        gt_lists[it->second].push_back(i);
    }

    std::vector<std::vector<int>> filtered_gt;
    filtered_gt.reserve(gt_lists.size());
    for (const auto& points : gt_lists) {
        if (static_cast<int>(points.size()) < args.min_gt_points) continue;
        const int gt_idx = static_cast<int>(filtered_gt.size());
        filtered_gt.push_back(points);
        for (int point_idx : points) point_to_gt[point_idx] = gt_idx;
    }

    std::ostringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());
    auto t0 = std::chrono::steady_clock::now();
    auto clusters = run_algorithm(frame.cloud, algorithm, args.min_cluster_size, args.tolerance, args.max_n);
    auto t1 = std::chrono::steady_clock::now();
    std::cout.rdbuf(old_buf);

    std::vector<double> best_iou(filtered_gt.size(), 0.0);
    std::vector<std::tuple<double, int, int>> candidate_matches;

    for (int pred_idx = 0; pred_idx < static_cast<int>(clusters.size()); ++pred_idx) {
        const auto& cluster = clusters[pred_idx];
        std::unordered_map<int, int> overlap_count;
        for (int point_idx : cluster.indices) {
            if (point_idx < 0 || point_idx >= static_cast<int>(point_to_gt.size())) continue;
            const int gt_idx = point_to_gt[point_idx];
            if (gt_idx >= 0) ++overlap_count[gt_idx];
        }
        for (const auto& [gt_idx, inter] : overlap_count) {
            const int gt_size = static_cast<int>(filtered_gt[gt_idx].size());
            const int pred_size = static_cast<int>(cluster.indices.size());
            const double union_size = static_cast<double>(gt_size + pred_size - inter);
            const double iou = union_size > 0.0 ? static_cast<double>(inter) / union_size : 0.0;
            best_iou[gt_idx] = std::max(best_iou[gt_idx], iou);
            if (iou >= 0.5) candidate_matches.emplace_back(iou, gt_idx, pred_idx);
        }
    }

    std::sort(candidate_matches.begin(), candidate_matches.end(),
        [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    std::vector<bool> gt_used(filtered_gt.size(), false);
    std::vector<bool> pred_used(clusters.size(), false);
    int tp = 0;
    double matched_iou_sum = 0.0;
    for (const auto& [iou, gt_idx, pred_idx] : candidate_matches) {
        if (gt_used[gt_idx] || pred_used[pred_idx]) continue;
        gt_used[gt_idx] = true;
        pred_used[pred_idx] = true;
        matched_iou_sum += iou;
        ++tp;
    }

    FrameMetrics metrics;
    metrics.raw_points = frame.raw_points;
    metrics.removed_ground = frame.removed_ground;
    metrics.points = static_cast<int>(frame.cloud->size());
    metrics.gt = static_cast<int>(filtered_gt.size());
    metrics.pred = static_cast<int>(clusters.size());
    metrics.tp = tp;
    metrics.fp = metrics.pred - tp;
    metrics.fn = metrics.gt - tp;
    metrics.matched_iou_sum = matched_iou_sum;
    metrics.best_iou_sum = std::accumulate(best_iou.begin(), best_iou.end(), 0.0);
    metrics.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return metrics;
}

static Summary evaluate_algorithm(
    const Args& args,
    const std::filesystem::path& sequence_root,
    const std::vector<std::string>& frame_ids,
    const std::string& algorithm
) {
    Summary summary;
    summary.frames = static_cast<int>(frame_ids.size());
    for (const auto& frame_id : frame_ids) {
        FrameMetrics r = evaluate_frame(args, sequence_root, frame_id, algorithm);
        summary.raw_points += r.raw_points;
        summary.removed_ground += r.removed_ground;
        summary.points += r.points;
        summary.gt += r.gt;
        summary.pred += r.pred;
        summary.tp += r.tp;
        summary.fp += r.fp;
        summary.fn += r.fn;
        summary.matched_iou_sum += r.matched_iou_sum;
        summary.best_iou_sum += r.best_iou_sum;
        summary.avg_time_ms += r.wall_ms;
        summary.avg_frame_fps += fps_from_ms(r.wall_ms);
    }
    if (!frame_ids.empty()) {
        const double denom = static_cast<double>(frame_ids.size());
        summary.avg_time_ms /= denom;
        summary.avg_frame_fps /= denom;
    }
    return summary;
}

static double sq(const Summary& s) {
    return s.tp > 0 ? s.matched_iou_sum / static_cast<double>(s.tp) : 0.0;
}

static double rq(const Summary& s) {
    const double denom = static_cast<double>(s.tp) + 0.5 * s.fp + 0.5 * s.fn;
    return denom > 0.0 ? static_cast<double>(s.tp) / denom : 0.0;
}

static double rc50(const Summary& s) {
    const int denom = s.tp + s.fn;
    return denom > 0 ? static_cast<double>(s.tp) / denom : 0.0;
}

static double miou(const Summary& s) {
    return s.gt > 0 ? s.best_iou_sum / static_cast<double>(s.gt) : 0.0;
}

static std::filesystem::path default_report_dir_from_argv0(const char* argv0) {
    std::filesystem::path exe_path = std::filesystem::weakly_canonical(std::filesystem::absolute(argv0));
    return exe_path.parent_path().parent_path().parent_path() / "reports";
}

static void write_report(
    const Args& args,
    const std::filesystem::path& path,
    const std::unordered_map<std::string, Summary>& summaries
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write report: " + path.string());

    out << "# SemanticPOSS Report-Style Benchmark\n\n";
    out << "## Setup\n\n";
    out << "- Dataset root: `" << args.root << "`\n";
    out << "- Sequence: `" << args.sequence << "`\n";
    out << "- Ground removal: `" << (args.remove_ground ? "semantic_id=22 removed" : "disabled") << "`\n";
    out << "- Parameters: `tol=" << args.tolerance << "`, `min_cluster_size=" << args.min_cluster_size
        << "`, `max_n=" << args.max_n << "`, `min_gt_points=" << args.min_gt_points << "`\n";
    out << "- FPS definition: `FPS = 1000 / frame_cluster_wall_ms`, then averaged over frames\n\n";

    out << "## Summary\n\n";
    out << "| Algorithm | Frames | Raw Points | Removed Ground | Points | GT | Pred | TP | FP | FN | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    out << std::fixed << std::setprecision(2);
    for (const auto& algorithm : args.algorithms) {
        const Summary& s = summaries.at(algorithm);
        const double sq_value = sq(s);
        const double rq_value = rq(s);
        out << "|" << algorithm
            << "|" << s.frames
            << "|" << s.raw_points
            << "|" << s.removed_ground
            << "|" << s.points
            << "|" << s.gt
            << "|" << s.pred
            << "|" << s.tp
            << "|" << s.fp
            << "|" << s.fn
            << "|" << s.avg_time_ms
            << "|" << s.avg_frame_fps
            << "|" << sq_value * rq_value * 100.0
            << "|" << sq_value * 100.0
            << "|" << rq_value * 100.0
            << "|" << rc50(s) * 100.0
            << "|" << miou(s) * 100.0
            << "|\n";
    }

    out << "\n## Metric Notes\n\n";
    out << "- `PQ = SQ * RQ`\n";
    out << "- `SQ` is the mean IoU of one-to-one matched instances with `IoU >= 0.5`.\n";
    out << "- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.\n";
    out << "- `RC50 = TP / (TP + FN)` with `IoU >= 0.5`.\n";
    out << "- `mIoU` is the mean best-overlap IoU over all GT instances.\n";
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 0;

    try {
        const std::filesystem::path root(args.root);
        const std::filesystem::path sequences_root = resolve_sequences_root(root);
        const std::filesystem::path sequence_root = sequences_root / args.sequence;
        const std::filesystem::path velodyne_dir = sequence_root / "velodyne";
        const std::filesystem::path labels_dir = sequence_root / "labels";
        if (!std::filesystem::exists(velodyne_dir) || !std::filesystem::exists(labels_dir)) {
            cerr << "SemanticPOSS sequence is incomplete: " << sequence_root << endl;
            return 1;
        }

        const std::vector<std::string> frame_ids = list_frame_ids(velodyne_dir, args);
        cout << "SemanticPOSS report-style eval\n";
        cout << "Root: " << root.string() << "\n";
        cout << "Sequence: " << args.sequence << "\n";
        cout << "Frames: " << frame_ids.size() << "\n";
        cout << "Remove ground: " << (args.remove_ground ? "yes" : "no") << "\n";

        std::unordered_map<std::string, Summary> summaries;
        for (const auto& algorithm : args.algorithms) {
            cout << "Running " << algorithm << " ..." << endl;
            Summary summary = evaluate_algorithm(args, sequence_root, frame_ids, algorithm);
            const double sq_value = sq(summary);
            const double rq_value = rq(summary);
            cout << std::fixed << std::setprecision(3)
                 << algorithm
                 << " time=" << summary.avg_time_ms << " ms"
                 << " fps=" << summary.avg_frame_fps
                 << " PQ=" << sq_value * rq_value * 100.0
                 << " SQ=" << sq_value * 100.0
                 << " RQ=" << rq_value * 100.0
                 << " RC50=" << rc50(summary) * 100.0
                 << " mIoU=" << miou(summary) * 100.0
                 << endl;
            summaries.emplace(algorithm, summary);
        }

        if (args.write_report) {
            const std::filesystem::path report_dir = args.report_dir.empty()
                ? default_report_dir_from_argv0(argv[0])
                : std::filesystem::path(args.report_dir);
            std::filesystem::create_directories(report_dir);
            const std::filesystem::path report_path =
                report_dir / ("semanticposs_report_style_seq" + args.sequence + ".md");
            write_report(args, report_path, summaries);
            cout << "Report written: " << report_path.string() << endl;
        }
    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }
    return 0;
}
