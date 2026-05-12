#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "../third_party/csf/src/CSF.h"
#include "FECunion.h"
#include "semantic_kitti_paths.h"

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

constexpr std::uint16_t kRoad = 40;
constexpr std::uint16_t kParking = 44;
constexpr std::uint16_t kSidewalk = 48;
constexpr std::uint16_t kOtherGround = 49;
constexpr std::uint16_t kLaneMarking = 60;
constexpr std::uint16_t kTerrain = 72;

struct Args {
    std::string mode;
    std::string root;
    std::string sequence = "00";
    std::string frame_id = "000000";
    std::filesystem::path input_ply;
    std::filesystem::path out_dir = "reports/snapshots";
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    bool csf_slope_smooth = true;
    double csf_time_step = 0.65;
    double csf_class_threshold = 0.5;
    double csf_cloth_resolution = 1.0;
    int csf_rigidness = 3;
    int csf_iterations = 500;
    std::string stem = "snapshot";
};

struct XYZI {
    float x;
    float y;
    float z;
    float intensity;
};

static void print_help(const char* argv0) {
    std::cout
        << "FECunion snapshot export\n"
        << "Usage:\n"
        << "  " << argv0 << " --mode MODE [options]\n\n"
        << "Modes:\n"
        << "  fusion-ply         load a fused ply from --input-ply\n"
        << "  semantic-frame     load seq/frame and remove ground by labels\n"
        << "  csf-frame          load seq/frame and remove ground by CSF\n";
}

static bool parse_bool01(const std::string& value) {
    if (value == "1" || value == "true" || value == "True") return true;
    if (value == "0" || value == "false" || value == "False") return false;
    throw std::runtime_error("Expected 0/1/true/false, got: " + value);
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (key == "--help") {
            print_help(argv[0]);
            return false;
        } else if (key == "--mode") {
            auto v = need(key); if (!v) return false; args.mode = *v;
        } else if (key == "--root") {
            auto v = need(key); if (!v) return false; args.root = *v;
        } else if (key == "--sequence") {
            auto v = need(key); if (!v) return false; args.sequence = *v;
        } else if (key == "--frame") {
            auto v = need(key); if (!v) return false; args.frame_id = *v;
        } else if (key == "--input-ply") {
            auto v = need(key); if (!v) return false; args.input_ply = *v;
        } else if (key == "--out-dir") {
            auto v = need(key); if (!v) return false; args.out_dir = *v;
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else if (key == "--csf-slope-smooth") {
            auto v = need(key); if (!v) return false; args.csf_slope_smooth = parse_bool01(*v);
        } else if (key == "--csf-time-step") {
            auto v = need(key); if (!v) return false; args.csf_time_step = std::stod(*v);
        } else if (key == "--csf-class-threshold") {
            auto v = need(key); if (!v) return false; args.csf_class_threshold = std::stod(*v);
        } else if (key == "--csf-cloth-resolution") {
            auto v = need(key); if (!v) return false; args.csf_cloth_resolution = std::stod(*v);
        } else if (key == "--csf-rigidness") {
            auto v = need(key); if (!v) return false; args.csf_rigidness = std::stoi(*v);
        } else if (key == "--csf-iterations") {
            auto v = need(key); if (!v) return false; args.csf_iterations = std::stoi(*v);
        } else if (key == "--stem") {
            auto v = need(key); if (!v) return false; args.stem = *v;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }
    if (args.mode.empty()) {
        std::cerr << "--mode is required\n";
        return false;
    }
    return true;
}

static bool is_ground_semantic(std::uint16_t semantic) {
    return semantic == kRoad ||
           semantic == kParking ||
           semantic == kSidewalk ||
           semantic == kOtherGround ||
           semantic == kLaneMarking ||
           semantic == kTerrain;
}

static std::vector<XYZI> load_bin(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open: " + path.string());
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<XYZI> raw(static_cast<std::size_t>(size / sizeof(XYZI)));
    in.read(reinterpret_cast<char*>(raw.data()), size);
    return raw;
}

static std::vector<std::uint32_t> load_labels(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open: " + path.string());
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> labels(static_cast<std::size_t>(size / sizeof(std::uint32_t)));
    in.read(reinterpret_cast<char*>(labels.data()), size);
    return labels;
}

static Cloud::Ptr load_semantic_frame(const Args& args) {
    const auto sequences_root = semantic_kitti_paths::resolve_sequences_root(
        args.root.empty() ? std::nullopt : std::optional<std::filesystem::path>(args.root)
    );
    const auto seq_root = sequences_root / args.sequence;
    const auto raw = load_bin(seq_root / "velodyne" / (args.frame_id + ".bin"));
    const auto labels = load_labels(seq_root / "labels" / (args.frame_id + ".label"));
    if (raw.size() != labels.size()) throw std::runtime_error("Point/label size mismatch");

    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const auto semantic = static_cast<std::uint16_t>(labels[i] & 0xffffu);
        if (is_ground_semantic(semantic)) continue;
        cloud->push_back(pcl::PointXYZ(raw[i].x, raw[i].y, raw[i].z));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static CSF make_csf(const Args& args) {
    CSF csf;
    csf.params.bSloopSmooth = args.csf_slope_smooth;
    csf.params.time_step = args.csf_time_step;
    csf.params.class_threshold = args.csf_class_threshold;
    csf.params.cloth_resolution = args.csf_cloth_resolution;
    csf.params.rigidness = args.csf_rigidness;
    csf.params.interations = args.csf_iterations;
    return csf;
}

static Cloud::Ptr load_csf_frame(const Args& args) {
    const auto sequences_root = semantic_kitti_paths::resolve_sequences_root(
        args.root.empty() ? std::nullopt : std::optional<std::filesystem::path>(args.root)
    );
    const auto seq_root = sequences_root / args.sequence;
    const auto raw = load_bin(seq_root / "velodyne" / (args.frame_id + ".bin"));

    std::vector<csf::Point> points;
    points.reserve(raw.size());
    for (const auto& p : raw) {
        points.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
    }

    auto csf = make_csf(args);
    csf.setPointCloud(points);
    std::vector<int> ground_indexes;
    std::vector<int> offground_indexes;
    csf.do_filtering(ground_indexes, offground_indexes, false);

    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(offground_indexes.size());
    for (const int idx : offground_indexes) {
        const auto& p = raw[static_cast<std::size_t>(idx)];
        cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static Cloud::Ptr load_fusion_ply(const Args& args) {
    auto cloud = std::make_shared<Cloud>();
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(args.input_ply.string(), *cloud) != 0) {
        throw std::runtime_error("Failed to load PLY: " + args.input_ply.string());
    }
    return cloud;
}

static pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorize_clusters(
    const Cloud::Ptr& cloud,
    const std::vector<pcl::PointIndices>& clusters
) {
    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> color(64, 255);

    for (const auto& cluster : clusters) {
        const std::uint8_t r = static_cast<std::uint8_t>(color(rng));
        const std::uint8_t g = static_cast<std::uint8_t>(color(rng));
        const std::uint8_t b = static_cast<std::uint8_t>(color(rng));
        for (const int idx : cluster.indices) {
            pcl::PointXYZRGB point;
            point.x = cloud->points[static_cast<std::size_t>(idx)].x;
            point.y = cloud->points[static_cast<std::size_t>(idx)].y;
            point.z = cloud->points[static_cast<std::size_t>(idx)].z;
            point.r = r;
            point.g = g;
            point.b = b;
            out->push_back(point);
        }
    }
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    out->is_dense = false;
    return out;
}

static void write_xyzrgb_csv(
    const std::filesystem::path& path,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud
) {
    std::ofstream out(path);
    out << "x,y,z,r,g,b\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& p : cloud->points) {
        out << p.x << ',' << p.y << ',' << p.z << ','
            << static_cast<int>(p.r) << ','
            << static_cast<int>(p.g) << ','
            << static_cast<int>(p.b) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        Cloud::Ptr cloud;
        if (args.mode == "fusion-ply") {
            cloud = load_fusion_ply(args);
        } else if (args.mode == "semantic-frame") {
            cloud = load_semantic_frame(args);
        } else if (args.mode == "csf-frame") {
            cloud = load_csf_frame(args);
        } else {
            throw std::runtime_error("Unknown mode: " + args.mode);
        }

        const auto clusters = FECunion(cloud, args.min_cluster_size, args.tolerance, args.max_n);
        const auto colored = colorize_clusters(cloud, clusters);

        std::filesystem::create_directories(args.out_dir);
        const auto ply_path = args.out_dir / (args.stem + "_fecunion_clusters.ply");
        const auto csv_path = args.out_dir / (args.stem + "_fecunion_clusters.csv");
        if (pcl::io::savePLYFileBinary(ply_path.string(), *colored) != 0) {
            throw std::runtime_error("Failed to save PLY: " + ply_path.string());
        }
        write_xyzrgb_csv(csv_path, colored);

        std::cout << "Input points: " << cloud->size() << "\n";
        std::cout << "Clustered points: " << colored->size() << "\n";
        std::cout << "Clusters: " << clusters.size() << "\n";
        std::cout << "PLY: " << ply_path << "\n";
        std::cout << "CSV: " << csv_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
