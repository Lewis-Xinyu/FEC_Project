#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "../third_party/csf/src/CSF.h"
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
    std::string root;
    std::string sequence = "00";
    std::string frame_id = "000000";
    std::filesystem::path out_dir = "reports/semantic_kitti_ground_preview";
    bool csf_slope_smooth = true;
    double csf_time_step = 0.65;
    double csf_class_threshold = 0.5;
    double csf_cloth_resolution = 1.0;
    int csf_rigidness = 3;
    int csf_iterations = 500;
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

static void print_help(const char* argv0) {
    std::cout
        << "SemanticKITTI ground-filter compare export\n"
        << "Usage:\n"
        << "  " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --root PATH\n"
        << "  --sequence ID          default 00\n"
        << "  --frame ID             default 000000\n"
        << "  --out-dir PATH\n"
        << "  --csf-slope-smooth 0|1\n"
        << "  --csf-time-step FLOAT\n"
        << "  --csf-class-threshold FLOAT\n"
        << "  --csf-cloth-resolution FLOAT\n"
        << "  --csf-rigidness N\n"
        << "  --csf-iterations N\n";
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
        } else if (key == "--root") {
            auto v = need(key); if (!v) return false; args.root = *v;
        } else if (key == "--sequence") {
            auto v = need(key); if (!v) return false; args.sequence = *v;
        } else if (key == "--frame") {
            auto v = need(key); if (!v) return false; args.frame_id = *v;
        } else if (key == "--out-dir") {
            auto v = need(key); if (!v) return false; args.out_dir = *v;
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
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            print_help(argv[0]);
            return false;
        }
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

static FrameData load_frame(const std::filesystem::path& bin_path, const std::filesystem::path& label_path) {
    FrameData frame;
    frame.raw = load_bin(bin_path);
    frame.labels = load_labels(label_path);
    if (frame.raw.size() != frame.labels.size()) {
        throw std::runtime_error("Point/label size mismatch");
    }
    return frame;
}

static Cloud::Ptr make_raw_cloud(const FrameData& frame) {
    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(frame.raw.size());
    for (const auto& p : frame.raw) {
        cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static Cloud::Ptr make_semantic_noground_cloud(const FrameData& frame) {
    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(frame.raw.size());
    for (std::size_t i = 0; i < frame.raw.size(); ++i) {
        const auto semantic = static_cast<std::uint16_t>(frame.labels[i] & 0xffffu);
        if (is_ground_semantic(semantic)) continue;
        const auto& p = frame.raw[i];
        cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
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

static Cloud::Ptr make_csf_noground_cloud(const FrameData& frame, const Args& args, int& removed_ground_points) {
    std::vector<csf::Point> points;
    points.reserve(frame.raw.size());
    for (const auto& p : frame.raw) {
        points.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
    }

    auto csf = make_csf(args);
    csf.setPointCloud(points);
    std::vector<int> ground_indexes;
    std::vector<int> offground_indexes;
    csf.do_filtering(ground_indexes, offground_indexes, false);

    removed_ground_points = static_cast<int>(ground_indexes.size());
    auto cloud = std::make_shared<Cloud>();
    cloud->reserve(offground_indexes.size());
    for (const int idx : offground_indexes) {
        const auto& p = frame.raw[static_cast<std::size_t>(idx)];
        cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static void save_cloud(const std::filesystem::path& path, const Cloud::Ptr& cloud) {
    if (pcl::io::savePLYFileBinary(path.string(), *cloud) != 0) {
        throw std::runtime_error("Failed to save PLY: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        const auto sequences_root = semantic_kitti_paths::resolve_sequences_root(
            args.root.empty() ? std::nullopt : std::optional<std::filesystem::path>(args.root)
        );
        const auto sequence_root = sequences_root / args.sequence;
        const auto frame = load_frame(
            sequence_root / "velodyne" / (args.frame_id + ".bin"),
            sequence_root / "labels" / (args.frame_id + ".label")
        );

        std::filesystem::create_directories(args.out_dir);
        const auto stem = "seq" + args.sequence + "_frame" + args.frame_id;

        auto raw_cloud = make_raw_cloud(frame);
        auto semantic_cloud = make_semantic_noground_cloud(frame);
        int csf_removed = 0;
        auto csf_cloud = make_csf_noground_cloud(frame, args, csf_removed);

        const auto raw_path = args.out_dir / (stem + "_raw.ply");
        const auto semantic_path = args.out_dir / (stem + "_semantic_noground.ply");
        const auto csf_path = args.out_dir / (stem + "_csf_noground.ply");

        save_cloud(raw_path, raw_cloud);
        save_cloud(semantic_path, semantic_cloud);
        save_cloud(csf_path, csf_cloud);

        std::cout << "Frame: seq=" << args.sequence << " frame=" << args.frame_id << "\n";
        std::cout << "Raw points: " << raw_cloud->size() << "\n";
        std::cout << "Semantic no-ground points: " << semantic_cloud->size()
                  << " removed=" << (raw_cloud->size() - semantic_cloud->size()) << "\n";
        std::cout << "CSF no-ground points: " << csf_cloud->size()
                  << " removed=" << csf_removed << "\n";
        std::cout << "Raw PLY: " << raw_path << "\n";
        std::cout << "Semantic PLY: " << semantic_path << "\n";
        std::cout << "CSF PLY: " << csf_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
