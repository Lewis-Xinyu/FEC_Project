#pragma warning(disable:4996)

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "FEC_profile.h"
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
    std::string mode = "single-frame";
    std::string root;
    std::string sequence = "00";
    std::string frame_id = "000000";
    std::filesystem::path fused_dir = "reports/semantic_kitti_seq00_static";
    int target_points = 0;
    std::vector<int> max_n_values = {0, 50};
    double tolerance = 0.2;
    int min_cluster_size = 100;
    std::filesystem::path out_dir = "reports/maxn_profile";
};

struct XYZI {
    float x;
    float y;
    float z;
    float intensity;
};

struct TargetMeta {
    int target = 0;
    std::filesystem::path ply_path;
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

static std::vector<int> parse_int_csv(const std::string& text) {
    std::vector<int> out;
    for (const auto& token : split_csv(text)) out.push_back(std::stoi(token));
    return out;
}

static void print_help(const char* argv0) {
    std::cout
        << "FEC max_n profile\n"
        << "Usage:\n"
        << "  " << argv0 << " [options]\n\n"
        << "Modes:\n"
        << "  --mode single-frame      Use one SemanticKITTI frame after semantic no-ground filtering\n"
        << "  --mode fusion            Use one fused PLY target from fusion_build output\n\n"
        << "Single-frame options:\n"
        << "  --root PATH              SemanticKITTI root\n"
        << "  --sequence ID            Sequence id, default 00\n"
        << "  --frame-id ID            Frame id, default 000000\n\n"
        << "Fusion options:\n"
        << "  --fused-dir PATH         Fusion build directory\n"
        << "  --target-points N        Target size to load from manifest\n\n"
        << "Shared options:\n"
        << "  --max-n-values LIST      Comma-separated list, default 0,50\n"
        << "  --tol FLOAT              Clustering tolerance, default 0.2\n"
        << "  --min-cluster-size N     Minimum cluster size, default 100\n"
        << "  --out-dir PATH           Output directory, default reports/maxn_profile\n";
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
        } else if (key == "--frame-id") {
            auto v = need(key); if (!v) return false; args.frame_id = *v;
        } else if (key == "--fused-dir") {
            auto v = need(key); if (!v) return false; args.fused_dir = *v;
        } else if (key == "--target-points") {
            auto v = need(key); if (!v) return false; args.target_points = std::stoi(*v);
        } else if (key == "--max-n-values") {
            auto v = need(key); if (!v) return false; args.max_n_values = parse_int_csv(*v);
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--out-dir") {
            auto v = need(key); if (!v) return false; args.out_dir = *v;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            print_help(argv[0]);
            return false;
        }
    }

    if (args.mode != "single-frame" && args.mode != "fusion") {
        std::cerr << "Unknown mode: " << args.mode << "\n";
        return false;
    }
    if (args.mode == "fusion" && args.target_points <= 0) {
        std::cerr << "--target-points is required in fusion mode\n";
        return false;
    }
    if (args.max_n_values.empty()) {
        std::cerr << "No max_n values provided\n";
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

static Cloud::Ptr load_single_frame_cloud(const Args& args) {
    const auto sequences_root = semantic_kitti_paths::resolve_sequences_root(
        args.root.empty() ? std::nullopt : std::optional<std::filesystem::path>(args.root)
    );
    const auto bin_path = sequences_root / args.sequence / "velodyne" / (args.frame_id + ".bin");
    const auto label_path = sequences_root / args.sequence / "labels" / (args.frame_id + ".label");
    const auto raw = load_bin(bin_path);
    const auto labels = load_labels(label_path);
    if (raw.size() != labels.size()) {
        throw std::runtime_error("Point/label size mismatch for " + bin_path.string());
    }

    auto cloud = Cloud::Ptr(new Cloud());
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

static std::filesystem::path resolve_path(
    const std::filesystem::path& raw_path,
    const std::filesystem::path& fused_dir
) {
    if (raw_path.empty()) throw std::runtime_error("Empty ply_path in manifest");
    if (raw_path.is_absolute() && std::filesystem::exists(raw_path)) return raw_path;
    if (std::filesystem::exists(raw_path)) return raw_path;
    const auto by_name = fused_dir / raw_path.filename();
    if (std::filesystem::exists(by_name)) return by_name;
    const auto by_join = fused_dir / raw_path;
    if (std::filesystem::exists(by_join)) return by_join;
    throw std::runtime_error("Cannot resolve PLY path: " + raw_path.string());
}

static TargetMeta load_target_meta(const Args& args) {
    const auto manifest = args.fused_dir / ("seq" + args.sequence + "_static_fusion_targets.csv");
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("Failed to open manifest: " + manifest.string());

    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < 9) continue;
        if (std::stoi(fields[0]) != args.target_points) continue;
        TargetMeta meta;
        meta.target = std::stoi(fields[0]);
        meta.ply_path = resolve_path(fields[8], args.fused_dir);
        return meta;
    }
    throw std::runtime_error("Target not found in manifest: " + std::to_string(args.target_points));
}

static Cloud::Ptr load_fusion_cloud(const Args& args) {
    const auto meta = load_target_meta(args);
    auto cloud = Cloud::Ptr(new Cloud());
    std::ostringstream captured;
    std::streambuf* old_err = std::cerr.rdbuf(captured.rdbuf());
    const int rc = pcl::io::loadPLYFile<pcl::PointXYZ>(meta.ply_path.string(), *cloud);
    std::cerr.rdbuf(old_err);
    if (rc != 0) throw std::runtime_error("Failed to load PLY: " + meta.ply_path.string());
    return cloud;
}

static std::string sample_name(const Args& args) {
    if (args.mode == "single-frame") {
        return "single_frame_seq" + args.sequence + "_" + args.frame_id;
    }
    return "fusion_seq" + args.sequence + "_" + std::to_string(args.target_points);
}

static void write_csv(
    const std::filesystem::path& path,
    const std::string& sample,
    const std::string& mode,
    const std::string& sequence,
    const std::string& frame_id,
    int target_points,
    const std::vector<FECProfileResult>& rows
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write CSV: " + path.string());

    out << "sample,mode,sequence,frame_id,target_points,points,max_n,min_cluster_size,tolerance,total_ms,build_ms,search_ms,merge_ms,final_ms,search_calls,avg_search_ms,total_neighbors_returned,avg_neighbors_returned,max_neighbors_returned,saturation_count,saturation_rate,neighbor_scan_pass1,neighbor_scan_pass2,relabel_trigger_count,relabel_point_visits,clusters,clustered_points\n";
    out << std::fixed << std::setprecision(6);

    for (const auto& row : rows) {
        const auto& s = row.stats;
        const double avg_search_ms = s.search_calls > 0 ? s.search_ms / static_cast<double>(s.search_calls) : 0.0;
        const double avg_neighbors = s.search_calls > 0 ? static_cast<double>(s.total_neighbors_returned) / static_cast<double>(s.search_calls) : 0.0;
        const double saturation_rate = s.search_calls > 0 ? static_cast<double>(s.saturation_count) / static_cast<double>(s.search_calls) : 0.0;
        out << sample << ','
            << mode << ','
            << sequence << ','
            << frame_id << ','
            << target_points << ','
            << s.points << ','
            << s.max_n << ','
            << s.min_cluster_size << ','
            << s.tolerance << ','
            << s.total_ms << ','
            << s.build_ms << ','
            << s.search_ms << ','
            << s.merge_ms << ','
            << s.final_ms << ','
            << s.search_calls << ','
            << avg_search_ms << ','
            << s.total_neighbors_returned << ','
            << avg_neighbors << ','
            << s.max_neighbors_returned << ','
            << s.saturation_count << ','
            << saturation_rate << ','
            << s.neighbor_scan_pass1 << ','
            << s.neighbor_scan_pass2 << ','
            << s.relabel_trigger_count << ','
            << s.relabel_point_visits << ','
            << s.clusters << ','
            << s.clustered_points << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        std::filesystem::create_directories(args.out_dir);

        Cloud::Ptr cloud;
        if (args.mode == "single-frame") {
            cloud = load_single_frame_cloud(args);
        } else {
            cloud = load_fusion_cloud(args);
        }

        std::vector<FECProfileResult> results;
        results.reserve(args.max_n_values.size());

        std::cout << "FEC max_n profile\n";
        std::cout << "Sample: " << sample_name(args) << "\n";
        std::cout << "Points: " << cloud->size() << "\n";

        for (int max_n : args.max_n_values) {
            auto profiled = FEC_profile(cloud, args.min_cluster_size, args.tolerance, max_n);
            results.push_back(profiled);
            const auto& s = profiled.stats;
            const double avg_search_ms = s.search_calls > 0 ? s.search_ms / static_cast<double>(s.search_calls) : 0.0;
            const double avg_neighbors = s.search_calls > 0 ? static_cast<double>(s.total_neighbors_returned) / static_cast<double>(s.search_calls) : 0.0;
            std::cout << std::fixed << std::setprecision(3)
                      << "  max_n=" << max_n
                      << " total=" << s.total_ms << " ms"
                      << " search_calls=" << s.search_calls
                      << " avg_search_ms=" << avg_search_ms
                      << " avg_neighbors=" << avg_neighbors
                      << " relabel_triggers=" << s.relabel_trigger_count
                      << " clusters=" << s.clusters << "\n";
        }

        const auto out_path = args.out_dir / (sample_name(args) + "_fec_maxn_profile.csv");
        write_csv(
            out_path,
            sample_name(args),
            args.mode,
            args.sequence,
            args.mode == "single-frame" ? args.frame_id : "",
            args.mode == "fusion" ? args.target_points : 0,
            results
        );
        std::cout << "CSV: " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
