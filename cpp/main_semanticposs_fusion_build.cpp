#pragma warning(disable:4996)

#include <algorithm>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "semantic_kitti_paths.h"

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

constexpr std::uint16_t kCar = 10;
constexpr std::uint16_t kBuilding = 50;
constexpr std::uint16_t kVegetation = 70;
constexpr std::uint16_t kTrunk = 71;
constexpr int kDiagPairs = 30;
constexpr int kDiagSamplePoints = 2500;

struct Args {
    std::string root;
    std::string sequence = "00";
    std::filesystem::path out_dir = "reports/semantic_kitti_seq00_static";
    std::vector<int> targets = {500000, 1000000, 3000000};
    double voxel_size = 0.20;
    double ghost_radius = 0.0;
    int max_frames = -1;
    bool write_ply = true;
};

struct XYZI {
    float x;
    float y;
    float z;
    float intensity;
};

struct FrameData {
    std::vector<XYZI> raw;
    std::vector<std::uint32_t> labels;
};

struct FusedPoint {
    pcl::PointXYZ point;
    std::uint16_t semantic = 0;
    int frame = -1;
};

struct BuildStats {
    long long raw_points = 0;
    long long static_points = 0;
    long long accepted_points = 0;
    long long ghost_rejected = 0;
    long long density_rejected = 0;
    int frames_processed = 0;
    int last_frame = -1;
};

struct FrameStats {
    std::string frame_id;
    int raw_points = 0;
    int static_points = 0;
    int accepted_points = 0;
    int ghost_rejected = 0;
    int density_rejected = 0;
    int fused_total = 0;
};

struct DistanceStats {
    double mean = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    int samples = 0;
};

struct AlignmentStats {
    int gap = 1;
    int pairs = 0;
    DistanceStats global;
    DistanceStats ego;
};

struct TargetStats {
    int target = 0;
    int frames_used = 0;
    std::unordered_map<std::uint16_t, int> counts;
    pcl::PointXYZ min_pt;
    pcl::PointXYZ max_pt;
    std::filesystem::path ply_path;
};

struct VoxelKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const {
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct SemanticVoxelKey {
    VoxelKey voxel;
    std::uint16_t semantic = 0;

    bool operator==(const SemanticVoxelKey& other) const {
        return semantic == other.semantic && voxel == other.voxel;
    }
};

struct SemanticVoxelKeyHash {
    std::size_t operator()(const SemanticVoxelKey& key) const {
        std::size_t h1 = VoxelKeyHash{}(key.voxel);
        std::size_t h2 = std::hash<std::uint16_t>{}(key.semantic);
        return h1 ^ (h2 << 1);
    }
};

enum class InsertResult {
    Accepted,
    GhostRejected,
    DensityRejected
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
        std::string spec = text.substr(6);
        std::vector<std::string> parts;
        std::stringstream ss(spec);
        std::string token;
        while (std::getline(ss, token, ':')) parts.push_back(trim(token));
        if (parts.size() != 3) throw std::runtime_error("range target format is range:start:end:step");
        int begin = std::stoi(parts[0]);
        int end = std::stoi(parts[1]);
        int step = std::stoi(parts[2]);
        for (int v = begin; v <= end; v += step) targets.push_back(v);
    } else {
        for (const auto& part : split_csv(text)) targets.push_back(std::stoi(part));
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
        } else if (key == "--voxel-size") {
            auto v = need(key); if (!v) return false; args.voxel_size = std::stod(*v);
        } else if (key == "--ghost-radius") {
            auto v = need(key); if (!v) return false; args.ghost_radius = std::stod(*v);
        } else if (key == "--max-frames") {
            auto v = need(key); if (!v) return false; args.max_frames = std::stoi(*v);
        } else if (key == "--write-ply") {
            auto v = need(key); if (!v) return false; args.write_ply = parse_bool(*v);
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }

    if (args.targets.empty()) {
        std::cerr << "At least one target is required\n";
        return false;
    }
    if (args.voxel_size <= 0.0) {
        std::cerr << "voxel_size must be > 0\n";
        return false;
    }
    if (args.ghost_radius < 0.0) {
        std::cerr << "ghost_radius must be >= 0\n";
        return false;
    }
    return true;
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

static FrameData load_frame(const std::filesystem::path& sequence_root, const std::string& frame_id) {
    FrameData frame;
    frame.raw = load_bin(sequence_root / "velodyne" / (frame_id + ".bin"));
    frame.labels = load_labels(sequence_root / "labels" / (frame_id + ".label"));
    if (frame.raw.size() != frame.labels.size()) {
        throw std::runtime_error("Point/label mismatch: " + frame_id);
    }
    return frame;
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

static bool include_static_semantic(std::uint16_t semantic) {
    return semantic == kCar || semantic == kBuilding || semantic == kVegetation || semantic == kTrunk;
}

static bool include_alignment_semantic(std::uint16_t semantic) {
    return semantic == kBuilding || semantic == kVegetation || semantic == kTrunk;
}

static int frame_index_of(const std::string& frame_id) {
    return std::stoi(frame_id);
}

static Eigen::Matrix4d frame_transform(
    int frame_index,
    const std::vector<Eigen::Matrix4d>& poses,
    const Eigen::Matrix4d& tr
) {
    if (frame_index >= 0 && frame_index < static_cast<int>(poses.size())) {
        return poses[frame_index] * tr;
    }
    return tr;
}

static pcl::PointXYZ transform_point(const XYZI& p, const Eigen::Matrix4d& transform) {
    Eigen::Vector4d q(p.x, p.y, p.z, 1.0);
    q = transform * q;
    return pcl::PointXYZ(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()));
}

static VoxelKey point_to_voxel(const pcl::PointXYZ& point, double voxel_size) {
    return VoxelKey{
        static_cast<int>(std::floor(point.x / voxel_size)),
        static_cast<int>(std::floor(point.y / voxel_size)),
        static_cast<int>(std::floor(point.z / voxel_size))
    };
}

static void finalize_cloud(Cloud& cloud) {
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;
}

static Cloud::Ptr load_filtered_frame_cloud(
    const std::filesystem::path& sequence_root,
    const std::string& frame_id,
    const Eigen::Matrix4d& transform,
    bool anchors_only
) {
    const FrameData frame = load_frame(sequence_root, frame_id);

    auto cloud = Cloud::Ptr(new Cloud());
    cloud->reserve(frame.raw.size());
    for (std::size_t i = 0; i < frame.raw.size(); ++i) {
        const std::uint16_t semantic = static_cast<std::uint16_t>(frame.labels[i] & 0xffffu);
        const bool keep = anchors_only ? include_alignment_semantic(semantic) : include_static_semantic(semantic);
        if (keep) cloud->push_back(transform_point(frame.raw[i], transform));
    }
    finalize_cloud(*cloud);
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
    const Cloud::Ptr& reference,
    const Cloud::Ptr& query,
    int max_samples,
    std::vector<double>& out
) {
    if (reference->empty() || query->empty()) return;

    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(reference);
    std::vector<int> indices(1);
    std::vector<float> squared(1);
    const int stride = std::max(1, static_cast<int>(query->size()) / std::max(1, max_samples));
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
    int gap
) {
    AlignmentStats stats;
    stats.gap = gap;
    if (gap <= 0 || static_cast<int>(frames.size()) <= gap) return stats;

    const int usable_pairs = std::min(kDiagPairs, static_cast<int>(frames.size()) - gap);
    std::vector<double> global_distances;
    std::vector<double> ego_distances;

    for (int i = 0; i < usable_pairs; ++i) {
        const int a = frame_index_of(frames[i]);
        const int b = frame_index_of(frames[i + gap]);
        const Eigen::Matrix4d ta = frame_transform(a, poses, tr);
        const Eigen::Matrix4d tb = frame_transform(b, poses, tr);

        auto global_a = load_filtered_frame_cloud(sequence_root, frames[i], ta, true);
        auto global_b = load_filtered_frame_cloud(sequence_root, frames[i + gap], tb, true);
        auto ego_a = load_filtered_frame_cloud(sequence_root, frames[i], Eigen::Matrix4d::Identity(), true);
        auto ego_b = load_filtered_frame_cloud(sequence_root, frames[i + gap], Eigen::Matrix4d::Identity(), true);

        append_nn_distances(global_a, global_b, kDiagSamplePoints, global_distances);
        append_nn_distances(ego_a, ego_b, kDiagSamplePoints, ego_distances);
        ++stats.pairs;
    }

    stats.global = summarize_distances(global_distances);
    stats.ego = summarize_distances(ego_distances);
    return stats;
}

class StaticFusionBuilder {
public:
    explicit StaticFusionBuilder(const Args& args)
        : voxel_size_(args.voxel_size),
          ghost_radius_(args.ghost_radius),
          ghost_radius_sq_(args.ghost_radius * args.ghost_radius),
          neighbor_steps_(std::max(0, static_cast<int>(std::ceil(args.ghost_radius / args.voxel_size)))) {}

    InsertResult insert(const pcl::PointXYZ& point, std::uint16_t semantic, int frame) {
        const VoxelKey key = point_to_voxel(point, voxel_size_);
        const SemanticVoxelKey semantic_key{key, semantic};

        if (ghost_radius_ > 0.0) {
            for (int dx = -neighbor_steps_; dx <= neighbor_steps_; ++dx) {
                for (int dy = -neighbor_steps_; dy <= neighbor_steps_; ++dy) {
                    for (int dz = -neighbor_steps_; dz <= neighbor_steps_; ++dz) {
                        VoxelKey neighbor{key.x + dx, key.y + dy, key.z + dz};
                        auto it = voxel_to_points_.find(neighbor);
                        if (it == voxel_to_points_.end()) continue;
                        for (int idx : it->second) {
                            const auto& existing = points_[idx];
                            if (existing.semantic != semantic) continue;
                            const double ddx = static_cast<double>(existing.point.x) - point.x;
                            const double ddy = static_cast<double>(existing.point.y) - point.y;
                            const double ddz = static_cast<double>(existing.point.z) - point.z;
                            const double dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                            if (dist_sq <= ghost_radius_sq_) return InsertResult::GhostRejected;
                        }
                    }
                }
            }
        }

        if (!occupied_voxels_.insert(semantic_key).second) return InsertResult::DensityRejected;

        points_.push_back(FusedPoint{point, semantic, frame});
        if (ghost_radius_ > 0.0) {
            voxel_to_points_[key].push_back(static_cast<int>(points_.size()) - 1);
        }
        return InsertResult::Accepted;
    }

    const std::vector<FusedPoint>& points() const {
        return points_;
    }

private:
    double voxel_size_ = 0.2;
    double ghost_radius_ = 0.0;
    double ghost_radius_sq_ = 0.0;
    int neighbor_steps_ = 0;
    std::vector<FusedPoint> points_;
    std::unordered_set<SemanticVoxelKey, SemanticVoxelKeyHash> occupied_voxels_;
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> voxel_to_points_;
};

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

static Cloud::Ptr prefix_cloud(const std::vector<FusedPoint>& points, int target) {
    auto cloud = Cloud::Ptr(new Cloud());
    const int n = std::min<int>(target, static_cast<int>(points.size()));
    cloud->reserve(n);
    for (int i = 0; i < n; ++i) cloud->push_back(points[i].point);
    finalize_cloud(*cloud);
    return cloud;
}

static int class_count(const TargetStats& stats, std::uint16_t semantic) {
    auto it = stats.counts.find(semantic);
    return it == stats.counts.end() ? 0 : it->second;
}

static int tree_count(const TargetStats& stats) {
    return class_count(stats, kVegetation) + class_count(stats, kTrunk);
}

static void record_insert_result(InsertResult result, FrameStats& row, BuildStats& build_stats) {
    if (result == InsertResult::Accepted) {
        ++row.accepted_points;
        ++build_stats.accepted_points;
    } else if (result == InsertResult::GhostRejected) {
        ++row.ghost_rejected;
        ++build_stats.ghost_rejected;
    } else {
        ++row.density_rejected;
        ++build_stats.density_rejected;
    }
}

static TargetStats make_target_stats(const std::vector<FusedPoint>& points, int target) {
    TargetStats stats;
    stats.target = std::min<int>(target, static_cast<int>(points.size()));
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
        const auto& fp = points[i];
        stats.min_pt.x = std::min(stats.min_pt.x, fp.point.x);
        stats.min_pt.y = std::min(stats.min_pt.y, fp.point.y);
        stats.min_pt.z = std::min(stats.min_pt.z, fp.point.z);
        stats.max_pt.x = std::max(stats.max_pt.x, fp.point.x);
        stats.max_pt.y = std::max(stats.max_pt.y, fp.point.y);
        stats.max_pt.z = std::max(stats.max_pt.z, fp.point.z);
        stats.counts[fp.semantic]++;
        max_frame = std::max(max_frame, fp.frame);
    }
    stats.frames_used = max_frame + 1;
    return stats;
}

static void write_frame_csv(
    const std::filesystem::path& out_dir,
    const std::string& sequence,
    const std::vector<FrameStats>& frame_stats
) {
    const auto path = out_dir / ("seq" + sequence + "_static_fusion_frames.csv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write frame CSV: " + path.string());
    out << "frame_id,raw_points,static_points,accepted_points,ghost_rejected,density_rejected,fused_total\n";
    for (const auto& row : frame_stats) {
        out << row.frame_id << ","
            << row.raw_points << ","
            << row.static_points << ","
            << row.accepted_points << ","
            << row.ghost_rejected << ","
            << row.density_rejected << ","
            << row.fused_total << "\n";
    }
}

static void write_target_csv(
    const Args& args,
    const std::vector<TargetStats>& target_stats
) {
    const auto path = args.out_dir / ("seq" + args.sequence + "_static_fusion_targets.csv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write target CSV: " + path.string());
    out << "target_points,frames_used,car,tree,building,bbox_x,bbox_y,bbox_z,ply_path\n";
    for (const auto& stats : target_stats) {
        out << stats.target << ","
            << stats.frames_used << ","
            << class_count(stats, kCar) << ","
            << tree_count(stats) << ","
            << class_count(stats, kBuilding) << ","
            << std::fixed << std::setprecision(3) << (stats.max_pt.x - stats.min_pt.x) << ","
            << (stats.max_pt.y - stats.min_pt.y) << ","
            << (stats.max_pt.z - stats.min_pt.z) << ","
            << stats.ply_path.string() << "\n";
    }
}

static void write_report(
    const Args& args,
    const std::filesystem::path& sequence_root,
    const std::vector<AlignmentStats>& alignments,
    const BuildStats& build_stats,
    const std::vector<TargetStats>& target_stats
) {
    const auto path = args.out_dir / ("seq" + args.sequence + "_static_fusion_report.md");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write report: " + path.string());

    out << "# SemanticKITTI seq" << args.sequence << " static fusion build\n\n";
    out << "## Setup\n\n";
    out << "- Sequence root: `" << sequence_root.string() << "`\n";
    out << "- Selected semantic ids: car=10, building=50, tree=vegetation(70)+trunk(71)\n";
    out << "- Transform: `global_point = pose(frame) * Tr * lidar_point`\n";
    out << "- Alignment diagnostic anchors: `tree + building` only\n";
    out << "- Downsample rule: keep original points only, at most one point per `(semantic, voxel)`\n";
    if (args.ghost_radius > 0.0) {
        out << "- Ghost rule: reject a new point when an earlier kept point of the same semantic already exists within `"
            << std::fixed << std::setprecision(3) << args.ghost_radius << " m`\n";
    } else {
        out << "- Ghost rule: disabled in the export path; alignment is checked by the diagnostic table instead\n";
    }
    out << "- Voxel size: `" << args.voxel_size << " m`\n";
    out << "- Frames processed: " << build_stats.frames_processed << "\n";
    out << "- Last frame used: " << build_stats.last_frame << "\n\n";

    out << "## Alignment Diagnostic\n\n";
    out << "Nearest-neighbor distances are computed on `tree/building` anchors only. "
        << "Lower global distances than ego distances indicate that pose alignment is working and large coordinate-misalignment ghosting is unlikely.\n\n";
    out << "| Frame gap | Space | Pairs | Samples | Mean m | Median m | P90 m | P95 m |\n";
    out << "|---:|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& alignment : alignments) {
        out << "|" << alignment.gap
            << "|pose global|"
            << alignment.pairs
            << "|" << alignment.global.samples
            << "|" << std::fixed << std::setprecision(3) << alignment.global.mean
            << "|" << alignment.global.median
            << "|" << alignment.global.p90
            << "|" << alignment.global.p95 << "|\n";
        out << "|" << alignment.gap
            << "|ego no pose|"
            << alignment.pairs
            << "|" << alignment.ego.samples
            << "|" << std::fixed << std::setprecision(3) << alignment.ego.mean
            << "|" << alignment.ego.median
            << "|" << alignment.ego.p90
            << "|" << alignment.ego.p95 << "|\n";
    }
    out << "\n";

    out << "## Build Summary\n\n";
    out << "| Raw points | Static candidates | Accepted | Ghost rejected | Density rejected |\n";
    out << "|---:|---:|---:|---:|---:|\n";
    out << "|" << build_stats.raw_points
        << "|" << build_stats.static_points
        << "|" << build_stats.accepted_points
        << "|" << build_stats.ghost_rejected
        << "|" << build_stats.density_rejected << "|\n\n";

    out << "## Exported Targets\n\n";
    out << "| Target points | Frames used | Car | Tree | Building | BBox X m | BBox Y m | BBox Z m | PLY |\n";
    out << "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
    for (const auto& stats : target_stats) {
        out << "|" << stats.target
            << "|" << stats.frames_used
            << "|" << class_count(stats, kCar)
            << "|" << tree_count(stats)
            << "|" << class_count(stats, kBuilding)
            << "|" << std::fixed << std::setprecision(3) << (stats.max_pt.x - stats.min_pt.x)
            << "|" << (stats.max_pt.y - stats.min_pt.y)
            << "|" << (stats.max_pt.z - stats.min_pt.z)
            << "|`" << stats.ply_path.string() << "`|\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        const std::filesystem::path seq_root =
            semantic_kitti_paths::resolve_sequences_root(
                args.root.empty() ? std::optional<std::filesystem::path>{}
                                  : std::optional<std::filesystem::path>{args.root}
            ) / args.sequence;
        const auto frames = list_frames(seq_root);
        const auto poses = load_poses(seq_root / "poses.txt");
        const Eigen::Matrix4d tr = load_tr(seq_root / "calib.txt");
        const int max_target = *std::max_element(args.targets.begin(), args.targets.end());
        const std::vector<int> diag_gaps = {1, 5, 10};

        std::filesystem::create_directories(args.out_dir);

        std::vector<AlignmentStats> alignments;
        alignments.reserve(diag_gaps.size());
        for (int gap : diag_gaps) {
            alignments.push_back(diagnose_alignment(seq_root, frames, poses, tr, gap));
        }

        StaticFusionBuilder builder(args);
        BuildStats build_stats;
        std::vector<FrameStats> frame_stats;

        std::cout << "Sequence root: " << seq_root << "\n";
        std::cout << "Building static fusion up to " << max_target << " kept points\n";

        for (const auto& frame_id : frames) {
            if (args.max_frames > 0 && build_stats.frames_processed >= args.max_frames) break;
            if (static_cast<int>(builder.points().size()) >= max_target) break;

            const int frame_index = frame_index_of(frame_id);
            const Eigen::Matrix4d transform = frame_transform(frame_index, poses, tr);
            const FrameData frame = load_frame(seq_root, frame_id);

            FrameStats row;
            row.frame_id = frame_id;
            row.raw_points = static_cast<int>(frame.raw.size());

            build_stats.raw_points += static_cast<long long>(frame.raw.size());
            build_stats.last_frame = frame_index;
            ++build_stats.frames_processed;

            for (std::size_t i = 0; i < frame.raw.size(); ++i) {
                const std::uint16_t semantic = static_cast<std::uint16_t>(frame.labels[i] & 0xffffu);
                if (!include_static_semantic(semantic)) continue;

                ++row.static_points;
                ++build_stats.static_points;

                const pcl::PointXYZ point = transform_point(frame.raw[i], transform);
                const InsertResult result = builder.insert(point, semantic, frame_index);
                record_insert_result(result, row, build_stats);

                if (static_cast<int>(builder.points().size()) >= max_target) break;
            }

            row.fused_total = static_cast<int>(builder.points().size());
            frame_stats.push_back(row);

            std::cout << "Frame " << frame_id
                      << " static=" << row.static_points
                      << " accepted=" << row.accepted_points
                      << " ghost=" << row.ghost_rejected
                      << " density=" << row.density_rejected
                      << " total=" << row.fused_total << "\n";
        }

        const auto& fused_points = builder.points();
        if (static_cast<int>(fused_points.size()) < max_target) {
            std::cerr << "Warning: only " << fused_points.size() << " kept points available\n";
        }

        std::vector<TargetStats> target_stats;
        target_stats.reserve(args.targets.size());
        for (int target : args.targets) {
            TargetStats stats = make_target_stats(fused_points, target);
            if (args.write_ply) {
                stats.ply_path = args.out_dir / ("seq" + args.sequence + "_static_" + target_name(stats.target) + ".ply");
                auto cloud = prefix_cloud(fused_points, stats.target);
                pcl::io::savePLYFileBinary(stats.ply_path.string(), *cloud);
            }
            target_stats.push_back(stats);
            std::cout << "Export target " << stats.target;
            if (!stats.ply_path.empty()) std::cout << " -> " << stats.ply_path;
            std::cout << "\n";
        }

        write_frame_csv(args.out_dir, args.sequence, frame_stats);
        write_target_csv(args, target_stats);
        write_report(args, seq_root, alignments, build_stats, target_stats);

        for (const auto& alignment : alignments) {
            std::cout << std::fixed << std::setprecision(3)
                      << "Alignment gap=" << alignment.gap
                      << " median global=" << alignment.global.median
                      << " m ego=" << alignment.ego.median << " m\n";
        }

        std::cout << "Report: " << (args.out_dir / ("seq" + args.sequence + "_static_fusion_report.md")) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
