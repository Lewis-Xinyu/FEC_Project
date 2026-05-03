#pragma warning(disable:4996)

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pcl/PointIndices.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "FEC.h"
#include "FECunion.h"

namespace {

struct Args {
    std::string sequence = "00";
    std::filesystem::path fused_dir = "reports/semantic_kitti_seq00_static";
    std::filesystem::path out_dir;
    std::vector<int> targets;
    std::vector<std::string> algorithms = {"FECunion", "FEC"};
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int repeat = 5;
    int warmup = 1;
};

struct TargetMeta {
    int target = 0;
    int frames_used = 0;
    int car = 0;
    int tree = 0;
    int building = 0;
    double bbox_x = 0.0;
    double bbox_y = 0.0;
    double bbox_z = 0.0;
    std::filesystem::path ply_path;
};

struct BenchStats {
    std::string algorithm;
    int target = 0;
    int points = 0;
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
    while (std::getline(ss, token, ',')) out.push_back(trim(token));
    return out;
}

static std::vector<int> parse_targets(const std::string& text) {
    std::vector<int> targets;
    if (text.rfind("range:", 0) == 0) {
        std::vector<std::string> parts;
        std::stringstream ss(text.substr(6));
        std::string token;
        while (std::getline(ss, token, ':')) parts.push_back(trim(token));
        if (parts.size() != 3) throw std::runtime_error("range target format is range:start:end:step");
        int begin = std::stoi(parts[0]);
        int end = std::stoi(parts[1]);
        int step = std::stoi(parts[2]);
        for (int v = begin; v <= end; v += step) targets.push_back(v);
    } else {
        for (const auto& token : split_csv(text)) {
            if (!token.empty()) targets.push_back(std::stoi(token));
        }
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    return targets;
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

        if (key == "--sequence") {
            auto v = need(key); if (!v) return false; args.sequence = *v;
        } else if (key == "--fused-dir") {
            auto v = need(key); if (!v) return false; args.fused_dir = *v;
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
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }

    if (args.out_dir.empty()) args.out_dir = args.fused_dir;
    if (args.repeat <= 0) args.repeat = 1;
    if (args.warmup < 0) args.warmup = 0;
    if (args.warmup >= args.repeat) args.warmup = args.repeat - 1;
    if (args.algorithms.empty()) {
        std::cerr << "No algorithms selected\n";
        return false;
    }
    return true;
}

static std::filesystem::path manifest_path(const Args& args) {
    return args.fused_dir / ("seq" + args.sequence + "_static_fusion_targets.csv");
}

static std::filesystem::path resolve_path(
    const std::filesystem::path& raw_path,
    const std::filesystem::path& fused_dir
) {
    if (raw_path.empty()) throw std::runtime_error("Empty ply_path in manifest");
    if (raw_path.is_absolute() && std::filesystem::exists(raw_path)) return raw_path;
    if (std::filesystem::exists(raw_path)) return raw_path;

    const std::filesystem::path by_name = fused_dir / raw_path.filename();
    if (std::filesystem::exists(by_name)) return by_name;

    const std::filesystem::path by_join = fused_dir / raw_path;
    if (std::filesystem::exists(by_join)) return by_join;

    throw std::runtime_error("Cannot resolve PLY path: " + raw_path.string());
}

static std::vector<TargetMeta> load_manifest(const Args& args) {
    const auto path = manifest_path(args);
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open manifest: " + path.string());

    std::string header;
    if (!std::getline(in, header)) throw std::runtime_error("Empty manifest: " + path.string());

    std::vector<TargetMeta> metas;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < 9) throw std::runtime_error("Malformed manifest row: " + line);

        TargetMeta meta;
        meta.target = std::stoi(fields[0]);
        meta.frames_used = std::stoi(fields[1]);
        meta.car = std::stoi(fields[2]);
        meta.tree = std::stoi(fields[3]);
        meta.building = std::stoi(fields[4]);
        meta.bbox_x = std::stod(fields[5]);
        meta.bbox_y = std::stod(fields[6]);
        meta.bbox_z = std::stod(fields[7]);
        meta.ply_path = resolve_path(fields[8], args.fused_dir);
        metas.push_back(meta);
    }

    std::sort(metas.begin(), metas.end(), [](const TargetMeta& a, const TargetMeta& b) {
        return a.target < b.target;
    });
    return metas;
}

static std::vector<TargetMeta> select_targets(
    const std::vector<TargetMeta>& metas,
    const std::vector<int>& requested
) {
    if (requested.empty()) return metas;

    std::unordered_map<int, TargetMeta> by_target;
    for (const auto& meta : metas) by_target[meta.target] = meta;

    std::vector<TargetMeta> out;
    out.reserve(requested.size());
    for (int target : requested) {
        auto it = by_target.find(target);
        if (it == by_target.end()) {
            throw std::runtime_error("Target not found in manifest: " + std::to_string(target));
        }
        out.push_back(it->second);
    }
    std::sort(out.begin(), out.end(), [](const TargetMeta& a, const TargetMeta& b) {
        return a.target < b.target;
    });
    return out;
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
    const TargetMeta& meta,
    const std::string& algorithm,
    const Args& args
) {
    using Clock = std::chrono::steady_clock;

    BenchStats stats;
    stats.algorithm = algorithm;
    stats.target = meta.target;
    stats.points = static_cast<int>(cloud->size());

    std::vector<double> kept;
    for (int run = 0; run < args.repeat; ++run) {
        std::ostringstream captured;
        std::streambuf* old = std::cout.rdbuf(captured.rdbuf());
        auto t0 = Clock::now();
        auto clusters = run_algorithm(cloud, algorithm, args);
        auto t1 = Clock::now();
        std::cout.rdbuf(old);

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (run >= args.warmup) kept.push_back(ms);
        stats.clusters = static_cast<int>(clusters.size());
    }

    if (kept.empty()) kept.push_back(0.0);
    stats.best_ms = *std::min_element(kept.begin(), kept.end());
    stats.mean_ms = std::accumulate(kept.begin(), kept.end(), 0.0) / kept.size();
    stats.fps = stats.best_ms > 1e-12 ? 1000.0 / stats.best_ms : 0.0;
    return stats;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_cloud(const std::filesystem::path& path) {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(path.string(), *cloud) != 0) {
        throw std::runtime_error("Failed to load PLY: " + path.string());
    }
    return cloud;
}

static void write_live_csv(
    const Args& args,
    const std::vector<BenchStats>& bench
) {
    const auto path = args.out_dir / ("seq" + args.sequence + "_static_timing_live.csv");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write timing CSV: " + path.string());

    out << "target_points,loaded_points,algorithm,best_ms,mean_ms,fps_best,clusters\n";
    for (const auto& row : bench) {
        out << row.target << ","
            << row.points << ","
            << row.algorithm << ","
            << std::fixed << std::setprecision(6) << row.best_ms << ","
            << row.mean_ms << ","
            << row.fps << ","
            << row.clusters << "\n";
    }
}

static void write_report(
    const Args& args,
    const std::vector<TargetMeta>& metas,
    const std::vector<BenchStats>& bench
) {
    const auto path = args.out_dir / ("seq" + args.sequence + "_static_timing_report.md");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write timing report: " + path.string());

    out << "# SemanticKITTI seq" << args.sequence << " static fusion timing report\n\n";
    out << "## Source\n\n";
    out << "- Fused directory: `" << args.fused_dir.string() << "`\n";
    out << "- Manifest: `" << manifest_path(args).string() << "`\n";
    out << "- Algorithms: `";
    for (std::size_t i = 0; i < args.algorithms.size(); ++i) {
        if (i) out << ", ";
        out << args.algorithms[i];
    }
    out << "`\n";
    out << "- Parameters: `tol=" << args.tolerance
        << ", min_cluster_size=" << args.min_cluster_size
        << ", max_n=" << args.max_n
        << ", repeat=" << args.repeat
        << ", warmup=" << args.warmup << "`\n\n";

    out << "## Loaded Fused Samples\n\n";
    out << "| Target points | Frames used | Car | Tree | Building | BBox X m | BBox Y m | BBox Z m | PLY |\n";
    out << "|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
    for (const auto& meta : metas) {
        out << "|" << meta.target
            << "|" << meta.frames_used
            << "|" << meta.car
            << "|" << meta.tree
            << "|" << meta.building
            << "|" << std::fixed << std::setprecision(3) << meta.bbox_x
            << "|" << meta.bbox_y
            << "|" << meta.bbox_z
            << "|`" << meta.ply_path.string() << "`|\n";
    }
    out << "\n";

    out << "## Timing\n\n";
    out << "| Target points | Loaded points | Algorithm | Best ms | Mean ms | FPS(best) | Clusters |\n";
    out << "|---:|---:|---|---:|---:|---:|---:|\n";
    for (const auto& row : bench) {
        out << "|" << row.target
            << "|" << row.points
            << "|" << row.algorithm
            << "|" << std::fixed << std::setprecision(3) << row.best_ms
            << "|" << row.mean_ms
            << "|" << row.fps
            << "|" << row.clusters << "|\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        std::filesystem::create_directories(args.out_dir);

        const auto metas = select_targets(load_manifest(args), args.targets);
        std::vector<BenchStats> bench;

        std::cout << "Fused directory: " << args.fused_dir << "\n";
        std::cout << "Loaded targets: " << metas.size() << "\n";

        for (const auto& meta : metas) {
            auto cloud = load_cloud(meta.ply_path);
            std::cout << "Target " << meta.target
                      << " points, file=" << meta.ply_path
                      << ", loaded=" << cloud->size() << "\n";
            for (const auto& algorithm : args.algorithms) {
                auto stats = benchmark_one(cloud, meta, algorithm, args);
                bench.push_back(stats);
                std::cout << std::fixed << std::setprecision(3)
                          << "  " << algorithm
                          << " best=" << stats.best_ms
                          << " ms fps=" << stats.fps
                          << " clusters=" << stats.clusters << "\n";
            }
        }

        write_live_csv(args, bench);
        write_report(args, metas, bench);
        std::cout << "Report: " << (args.out_dir / ("seq" + args.sequence + "_static_timing_report.md")) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
