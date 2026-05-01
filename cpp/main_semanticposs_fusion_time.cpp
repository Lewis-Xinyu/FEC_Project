#pragma warning(disable:4996)

#include <algorithm>
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
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <pcl/PointIndices.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "FEC.h"
#include "FECunion.h"

namespace {

constexpr std::uint16_t kCar = 7;
constexpr std::uint16_t kTrunk = 8;
constexpr std::uint16_t kPlants = 9;
constexpr std::uint16_t kBuilding = 15;

struct Args {
    std::string root = "/mnt/d/semanticposs";
    std::string sequence = "00";
    std::filesystem::path out_dir = "reports/fusion_seq00";
    std::vector<int> targets = {50000, 500000, 1000000, 3000000};
    std::vector<std::string> algorithms = {"FECunion", "FEC"};
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int repeat = 5;
    int warmup = 1;
    int diag_pairs = 30;
    std::vector<int> diag_gaps = {1, 5, 10};
    int diag_sample_points = 2500;
    bool write_ply = true;
    bool benchmark = false;
};

struct XYZI {
    float x;
    float y;
    float z;
    float intensity;
};

struct FusedData {
    std::vector<pcl::PointXYZ> points;
    std::vector<std::uint16_t> semantics;
    std::vector<int> frames;
    int last_frame = -1;
    long long raw_points = 0;
};

struct TargetStats {
    int target = 0;
    int frames_used = 0;
    std::unordered_map<std::uint16_t, int> counts;
    pcl::PointXYZ min_pt;
    pcl::PointXYZ max_pt;
    std::filesystem::path ply_path;
};

struct DistanceStats {
    double mean = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    int samples = 0;
};

struct AlignmentStats {
    DistanceStats global;
    DistanceStats ego;
    int gap = 1;
    int pairs = 0;
};

struct BenchStats {
    std::string algorithm;
    int target = 0;
    int clusters = 0;
    double best_ms = 0.0;
    double mean_ms = 0.0;
    double fps = 0.0;
};

static std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
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

static std::vector<int> parse_targets(const std::string& text) {
    std::vector<int> targets;
    if (text.rfind("range:", 0) == 0) {
        std::vector<std::string> parts = split_csv(text.substr(6));
        if (parts.size() == 1) {
            std::stringstream ss(parts.front());
            std::string token;
            parts.clear();
            while (std::getline(ss, token, ':')) parts.push_back(trim(token));
        }
        if (parts.size() != 3) throw std::runtime_error("range target format is range:start:end:step");
        int begin = std::stoi(parts[0]);
        int end = std::stoi(parts[1]);
        int step = std::stoi(parts[2]);
        for (int v = begin; v <= end; v += step) targets.push_back(v);
    } else {
        for (const auto& token : split_csv(text)) targets.push_back(std::stoi(token));
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    return targets;
}

static bool parse_bool(const std::string& value) {
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };
        if (key == "--root") {
            auto v = need(key); if (!v) return false; args.root = *v;
        } else if (key == "--sequence") {
            auto v = need(key); if (!v) return false; args.sequence = *v;
        } else if (key == "--out-dir") {
            auto v = need(key); if (!v) return false; args.out_dir = *v;
        } else if (key == "--targets") {
            auto v = need(key); if (!v) return false; args.targets = parse_targets(*v);
        } else if (key == "--alg") {
            auto v = need(key); if (!v) return false; args.algorithms = split_csv(*v);
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else if (key == "--repeat") {
            auto v = need(key); if (!v) return false; args.repeat = std::stoi(*v);
        } else if (key == "--warmup") {
            auto v = need(key); if (!v) return false; args.warmup = std::stoi(*v);
        } else if (key == "--diag-pairs") {
            auto v = need(key); if (!v) return false; args.diag_pairs = std::stoi(*v);
        } else if (key == "--diag-gaps") {
            auto v = need(key); if (!v) return false; args.diag_gaps = parse_targets(*v);
        } else if (key == "--diag-sample-points") {
            auto v = need(key); if (!v) return false; args.diag_sample_points = std::stoi(*v);
        } else if (key == "--write-ply") {
            auto v = need(key); if (!v) return false; args.write_ply = parse_bool(*v);
        } else if (key == "--benchmark") {
            auto v = need(key); if (!v) return false; args.benchmark = parse_bool(*v);
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }
    return !args.targets.empty();
}

static std::filesystem::path sequences_root(const std::filesystem::path& root) {
    if (std::filesystem::exists(root / "sequences")) return root / "sequences";
    if (std::filesystem::exists(root / "dataset" / "sequences")) return root / "dataset" / "sequences";
    throw std::runtime_error("SemanticPOSS root is incomplete: " + root.string());
}

static std::vector<std::string> list_frames(const std::filesystem::path& sequence_root) {
    std::vector<std::string> frames;
    for (const auto& entry : std::filesystem::directory_iterator(sequence_root / "velodyne")) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            frames.push_back(entry.path().stem().string());
        }
    }
    std::sort(frames.begin(), frames.end());
    return frames;
}

static std::vector<XYZI> load_bin(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open point cloud: " + path.string());
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<XYZI> raw(static_cast<std::size_t>(size / sizeof(XYZI)));
    in.read(reinterpret_cast<char*>(raw.data()), size);
    return raw;
}

static std::vector<std::uint32_t> load_labels(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open label file: " + path.string());
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> labels(static_cast<std::size_t>(size / sizeof(std::uint32_t)));
    in.read(reinterpret_cast<char*>(labels.data()), size);
    return labels;
}

static std::vector<Eigen::Matrix4d> load_poses(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open poses: " + path.string());
    std::vector<Eigen::Matrix4d> poses;
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) ss >> pose(r, c);
        }
        poses.push_back(pose);
    }
    return poses;
}

static Eigen::Matrix4d load_tr(const std::filesystem::path& path) {
    std::ifstream in(path);
    Eigen::Matrix4d tr = Eigen::Matrix4d::Identity();
    if (!in) return tr;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("Tr:", 0) != 0) continue;
        std::stringstream ss(line.substr(3));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) ss >> tr(r, c);
        }
    }
    return tr;
}

static bool include_semantic(std::uint16_t semantic, bool static_only = false) {
    if (static_only) return semantic == kTrunk || semantic == kPlants || semantic == kBuilding;
    return semantic == kCar || semantic == kTrunk || semantic == kPlants || semantic == kBuilding;
}

static pcl::PointXYZ transform_point(const XYZI& p, const Eigen::Matrix4d& transform) {
    Eigen::Vector4d q(p.x, p.y, p.z, 1.0);
    q = transform * q;
    return pcl::PointXYZ(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()));
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_filtered_frame(
    const std::filesystem::path& sequence_root,
    const std::string& frame_id,
    const Eigen::Matrix4d& transform,
    bool static_only
) {
    auto raw = load_bin(sequence_root / "velodyne" / (frame_id + ".bin"));
    auto labels = load_labels(sequence_root / "labels" / (frame_id + ".label"));
    if (raw.size() != labels.size()) throw std::runtime_error("Point/label mismatch: " + frame_id);
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        std::uint16_t semantic = static_cast<std::uint16_t>(labels[i] & 0xffffu);
        if (include_semantic(semantic, static_only)) cloud->push_back(transform_point(raw[i], transform));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static DistanceStats summarize_distances(std::vector<double>& distances) {
    DistanceStats stats;
    stats.samples = static_cast<int>(distances.size());
    if (distances.empty()) return stats;
    std::sort(distances.begin(), distances.end());
    stats.mean = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    stats.median = distances[distances.size() / 2];
    stats.p90 = distances[static_cast<std::size_t>(std::floor((distances.size() - 1) * 0.90))];
    stats.p95 = distances[static_cast<std::size_t>(std::floor((distances.size() - 1) * 0.95))];
    return stats;
}

static void append_nn_distances(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& reference,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& query,
    int max_samples,
    std::vector<double>& out
) {
    if (reference->empty() || query->empty()) return;
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(reference);
    std::vector<int> indices(1);
    std::vector<float> squared(1);
    int stride = std::max(1, static_cast<int>(query->size()) / std::max(1, max_samples));
    for (int i = 0; i < static_cast<int>(query->size()); i += stride) {
        if (tree.nearestKSearch(query->points[i], 1, indices, squared) > 0) {
            out.push_back(std::sqrt(static_cast<double>(squared.front())));
        }
    }
}

static AlignmentStats diagnose_alignment(
    const std::filesystem::path& sequence_root,
    const std::vector<std::string>& frames,
    const std::vector<Eigen::Matrix4d>& poses,
    const Eigen::Matrix4d& tr,
    int pairs,
    int gap,
    int max_samples
) {
    std::vector<double> global_distances;
    std::vector<double> ego_distances;
    AlignmentStats stats;
    stats.gap = gap;
    int usable_pairs = std::min<int>(pairs, static_cast<int>(frames.size()) - gap);
    for (int i = 0; i < usable_pairs; ++i) {
        int a = std::stoi(frames[i]);
        int b = std::stoi(frames[i + gap]);
        Eigen::Matrix4d ta = (a >= 0 && a < static_cast<int>(poses.size())) ? poses[a] * tr : tr;
        Eigen::Matrix4d tb = (b >= 0 && b < static_cast<int>(poses.size())) ? poses[b] * tr : tr;
        auto global_a = load_filtered_frame(sequence_root, frames[i], ta, true);
        auto global_b = load_filtered_frame(sequence_root, frames[i + gap], tb, true);
        auto ego_a = load_filtered_frame(sequence_root, frames[i], Eigen::Matrix4d::Identity(), true);
        auto ego_b = load_filtered_frame(sequence_root, frames[i + gap], Eigen::Matrix4d::Identity(), true);
        append_nn_distances(global_a, global_b, max_samples, global_distances);
        append_nn_distances(ego_a, ego_b, max_samples, ego_distances);
        ++stats.pairs;
    }
    stats.global = summarize_distances(global_distances);
    stats.ego = summarize_distances(ego_distances);
    return stats;
}

static FusedData build_fused(
    const std::filesystem::path& sequence_root,
    const std::vector<std::string>& frames,
    const std::vector<Eigen::Matrix4d>& poses,
    const Eigen::Matrix4d& tr,
    int max_points
) {
    FusedData fused;
    fused.points.reserve(max_points);
    fused.semantics.reserve(max_points);
    fused.frames.reserve(max_points);
    for (const auto& frame_id : frames) {
        int frame_index = std::stoi(frame_id);
        Eigen::Matrix4d transform = tr;
        if (frame_index >= 0 && frame_index < static_cast<int>(poses.size())) {
            transform = poses[frame_index] * tr;
        }
        auto raw = load_bin(sequence_root / "velodyne" / (frame_id + ".bin"));
        auto labels = load_labels(sequence_root / "labels" / (frame_id + ".label"));
        if (raw.size() != labels.size()) throw std::runtime_error("Point/label mismatch: " + frame_id);
        fused.raw_points += static_cast<long long>(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            std::uint16_t semantic = static_cast<std::uint16_t>(labels[i] & 0xffffu);
            if (!include_semantic(semantic)) continue;
            fused.points.push_back(transform_point(raw[i], transform));
            fused.semantics.push_back(semantic);
            fused.frames.push_back(frame_index);
            if (static_cast<int>(fused.points.size()) >= max_points) {
                fused.last_frame = frame_index;
                return fused;
            }
        }
        fused.last_frame = frame_index;
    }
    return fused;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr prefix_cloud(const FusedData& fused, int target) {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    int n = std::min<int>(target, static_cast<int>(fused.points.size()));
    cloud->points.assign(fused.points.begin(), fused.points.begin() + n);
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static TargetStats make_target_stats(const FusedData& fused, int target) {
    TargetStats stats;
    stats.target = std::min<int>(target, static_cast<int>(fused.points.size()));
    stats.min_pt = pcl::PointXYZ(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    stats.max_pt = pcl::PointXYZ(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    int max_frame = -1;
    for (int i = 0; i < stats.target; ++i) {
        const auto& p = fused.points[i];
        stats.min_pt.x = std::min(stats.min_pt.x, p.x);
        stats.min_pt.y = std::min(stats.min_pt.y, p.y);
        stats.min_pt.z = std::min(stats.min_pt.z, p.z);
        stats.max_pt.x = std::max(stats.max_pt.x, p.x);
        stats.max_pt.y = std::max(stats.max_pt.y, p.y);
        stats.max_pt.z = std::max(stats.max_pt.z, p.z);
        stats.counts[fused.semantics[i]]++;
        max_frame = std::max(max_frame, fused.frames[i]);
    }
    stats.frames_used = max_frame + 1;
    return stats;
}

static std::vector<pcl::PointIndices> run_algorithm(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::string& algorithm,
    const Args& args
) {
    if (algorithm == "FECunion") return FECunion(cloud, args.min_cluster_size, args.tolerance, args.max_n);
    if (algorithm == "FEC") return FEC(cloud, args.min_cluster_size, args.tolerance, args.max_n);
    throw std::runtime_error("Unknown algorithm: " + algorithm);
}

static BenchStats benchmark_one(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    int target,
    const std::string& algorithm,
    const Args& args
) {
    using Clock = std::chrono::steady_clock;
    BenchStats stats;
    stats.algorithm = algorithm;
    stats.target = target;
    std::vector<double> kept;
    for (int run = 0; run < args.repeat; ++run) {
        std::ostringstream captured;
        std::streambuf* old = std::cout.rdbuf(captured.rdbuf());
        auto t0 = Clock::now();
        auto clusters = run_algorithm(cloud, algorithm, args);
        auto t1 = Clock::now();
        std::cout.rdbuf(old);
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (run >= args.warmup) kept.push_back(ms);
        stats.clusters = static_cast<int>(clusters.size());
    }
    if (kept.empty()) kept.push_back(0.0);
    stats.best_ms = *std::min_element(kept.begin(), kept.end());
    stats.mean_ms = std::accumulate(kept.begin(), kept.end(), 0.0) / kept.size();
    stats.fps = stats.best_ms > 1e-12 ? 1000.0 / stats.best_ms : 0.0;
    return stats;
}

static int class_count(const TargetStats& stats, std::uint16_t semantic) {
    auto it = stats.counts.find(semantic);
    return it == stats.counts.end() ? 0 : it->second;
}

static void write_report(
    const Args& args,
    const std::filesystem::path& sequence_root,
    const std::vector<AlignmentStats>& alignments,
    const FusedData& fused,
    const std::vector<TargetStats>& target_stats,
    const std::vector<BenchStats>& bench
) {
    std::filesystem::create_directories(args.out_dir);
    std::filesystem::path path = args.out_dir / ("seq" + args.sequence + "_fusion_report.md");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write report: " + path.string());
    out << "# SemanticPOSS seq" << args.sequence << " fusion report\n\n";
    out << "- Sequence root: `" << sequence_root.string() << "`\n";
    out << "- Selected semantic ids: car=7, tree=8(trunk)+9(plants), building=15\n";
    out << "- Transform: `global_point = pose(frame) * Tr * lidar_point`; seq" << args.sequence
        << " `Tr` is read from `calib.txt`.\n";
    out << "- Raw points loaded until max target: " << fused.raw_points << "\n";
    out << "- Fused selected points available: " << fused.points.size() << "\n";
    out << "- Last frame used: " << fused.last_frame << "\n\n";

    out << "## Alignment diagnostic\n\n";
    out << "Nearest-neighbor distances are computed between consecutive frames on static classes only "
        << "(tree/building), comparing pose-aligned global coordinates against unaligned ego coordinates.\n\n";
    out << "| Frame gap | Space | Pairs | Samples | Mean m | Median m | P90 m | P95 m |\n";
    out << "|---:|---|---:|---:|---:|---:|---:|---:|\n";
    auto row = [&](int gap, int pairs, const std::string& name, const DistanceStats& s) {
        out << "|" << gap << "|" << name << "|" << pairs << "|" << s.samples
            << "|" << std::fixed << std::setprecision(3) << s.mean
            << "|" << s.median << "|" << s.p90 << "|" << s.p95 << "|\n";
    };
    for (const auto& alignment : alignments) {
        row(alignment.gap, alignment.pairs, "pose global", alignment.global);
        row(alignment.gap, alignment.pairs, "ego no pose", alignment.ego);
    }
    out << "\n";

    out << "## Generated samples\n\n";
    out << "| Target points | Frames used | Car | Tree | Building | BBox X m | BBox Y m | BBox Z m | PLY |\n";
    out << "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
    for (const auto& s : target_stats) {
        int car = class_count(s, kCar);
        int tree = class_count(s, kTrunk) + class_count(s, kPlants);
        int building = class_count(s, kBuilding);
        out << "|" << s.target
            << "|" << s.frames_used
            << "|" << car
            << "|" << tree
            << "|" << building
            << "|" << std::fixed << std::setprecision(1) << (s.max_pt.x - s.min_pt.x)
            << "|" << (s.max_pt.y - s.min_pt.y)
            << "|" << (s.max_pt.z - s.min_pt.z)
            << "|`" << s.ply_path.string() << "`|\n";
    }
    out << "\n";

    if (!bench.empty()) {
        out << "## Timing\n\n";
        out << "Best time is the minimum after warmup. repeat=" << args.repeat
            << ", warmup=" << args.warmup << ", tol=" << args.tolerance
            << ", min_cluster_size=" << args.min_cluster_size << ", max_n=" << args.max_n << ".\n\n";
        out << "| Target points | Algorithm | Best ms | Mean ms | FPS(best) | Clusters |\n";
        out << "|---:|---|---:|---:|---:|---:|\n";
        for (const auto& b : bench) {
            out << "|" << b.target << "|" << b.algorithm
                << "|" << std::fixed << std::setprecision(3) << b.best_ms
                << "|" << b.mean_ms
                << "|" << b.fps
                << "|" << b.clusters << "|\n";
        }
        out << "\n";
    }
}

static std::string target_name(int target) {
    std::ostringstream ss;
    if (target % 1000000 == 0) {
        ss << (target / 1000000) << "M";
    } else if (target % 1000 == 0) {
        ss << (target / 1000) << "k";
    } else {
        ss << target;
    }
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        std::filesystem::path seq_root = sequences_root(args.root) / args.sequence;
        std::filesystem::create_directories(args.out_dir);
        auto frames = list_frames(seq_root);
        auto poses = load_poses(seq_root / "poses.txt");
        Eigen::Matrix4d tr = load_tr(seq_root / "calib.txt");
        int max_target = *std::max_element(args.targets.begin(), args.targets.end());

        std::cout << "Sequence root: " << seq_root << "\n";
        std::cout << "Frames: " << frames.size() << " poses: " << poses.size() << "\n";
        std::cout << "Building fusion up to " << max_target << " selected points\n";

        std::vector<AlignmentStats> alignments;
        for (int gap : args.diag_gaps) {
            if (gap > 0) {
                alignments.push_back(diagnose_alignment(seq_root, frames, poses, tr, args.diag_pairs, gap, args.diag_sample_points));
            }
        }
        auto fused = build_fused(seq_root, frames, poses, tr, max_target);
        if (static_cast<int>(fused.points.size()) < max_target) {
            std::cerr << "Warning: only " << fused.points.size() << " selected points available\n";
        }

        std::ofstream live_csv;
        if (args.benchmark) {
            live_csv.open(args.out_dir / ("seq" + args.sequence + "_fusion_timing_live.csv"));
            if (!live_csv) throw std::runtime_error("Failed to write live timing CSV");
            live_csv << "target_points,algorithm,best_ms,mean_ms,fps_best,clusters\n";
        }

        std::vector<TargetStats> target_stats;
        std::vector<BenchStats> bench;
        for (int target : args.targets) {
            auto cloud = prefix_cloud(fused, target);
            TargetStats stats = make_target_stats(fused, target);
            if (args.write_ply) {
                stats.ply_path = args.out_dir / ("seq" + args.sequence + "_tree_building_car_" + target_name(stats.target) + ".ply");
                pcl::io::savePLYFileBinary(stats.ply_path.string(), *cloud);
            }
            target_stats.push_back(stats);
            std::cout << "Prepared target " << stats.target << " points";
            if (!stats.ply_path.empty()) std::cout << " -> " << stats.ply_path;
            std::cout << "\n";
            if (args.benchmark) {
                for (const auto& algorithm : args.algorithms) {
                    auto b = benchmark_one(cloud, stats.target, algorithm, args);
                    bench.push_back(b);
                    live_csv << b.target << "," << b.algorithm << ","
                             << std::fixed << std::setprecision(6) << b.best_ms << ","
                             << b.mean_ms << "," << b.fps << "," << b.clusters << "\n";
                    live_csv.flush();
                    std::cout << std::fixed << std::setprecision(3)
                              << "  " << algorithm << " best=" << b.best_ms
                              << " ms fps=" << b.fps << " clusters=" << b.clusters << "\n";
                }
            }
        }

        write_report(args, seq_root, alignments, fused, target_stats, bench);
        std::cout << "Report: " << (args.out_dir / ("seq" + args.sequence + "_fusion_report.md")) << "\n";
        for (const auto& alignment : alignments) {
            std::cout << std::fixed << std::setprecision(3)
                      << "Alignment gap=" << alignment.gap
                      << " median global=" << alignment.global.median
                      << " m ego=" << alignment.ego.median << " m\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
