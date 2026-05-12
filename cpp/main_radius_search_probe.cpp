#pragma warning(disable:4996)

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace {

struct Args {
    std::filesystem::path ply_path;
    double tolerance = 0.2;
    int sample_count = 10000;
    int preview = 12;
    int max_n = 50;
};

static void print_help(const char* argv0) {
    std::cout
        << "Radius-search probe\n"
        << "Usage:\n"
        << "  " << argv0 << " --ply PATH [--tol 0.2] [--sample-count 10000] [--preview 12] [--max-n 50]\n";
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
        } else if (key == "--ply") {
            auto v = need(key); if (!v) return false; args.ply_path = *v;
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--sample-count") {
            auto v = need(key); if (!v) return false; args.sample_count = std::stoi(*v);
        } else if (key == "--preview") {
            auto v = need(key); if (!v) return false; args.preview = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            return false;
        }
    }
    if (args.ply_path.empty()) {
        std::cerr << "--ply is required\n";
        return false;
    }
    return true;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_cloud(const std::filesystem::path& path) {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    std::ostringstream captured;
    std::streambuf* old_err = std::cerr.rdbuf(captured.rdbuf());
    const int rc = pcl::io::loadPLYFile<pcl::PointXYZ>(path.string(), *cloud);
    std::cerr.rdbuf(old_err);
    if (rc != 0) throw std::runtime_error("Failed to load PLY: " + path.string());
    return cloud;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        const auto cloud = load_cloud(args.ply_path);
        pcl::KdTreeFLANN<pcl::PointXYZ> tree;
        tree.setInputCloud(cloud);

        const int total_points = static_cast<int>(cloud->size());
        const int sample_count = std::min(args.sample_count, total_points);
        const int stride = std::max(1, total_points / std::max(1, sample_count));

        std::vector<int> idx0;
        std::vector<float> d20;
        std::vector<int> idxn;
        std::vector<float> d2n;

        int sampled = 0;
        int capped_count = 0;
        int over_limit_without_cap = 0;
        std::int64_t sum0 = 0;
        std::int64_t sumn = 0;
        int max0 = 0;
        int maxn = 0;

        std::cout << "PLY: " << args.ply_path << "\n";
        std::cout << "Points: " << total_points << " tol=" << args.tolerance << " max_n=" << args.max_n << "\n";
        std::cout << "Preview of neighbor counts:\n";

        for (int i = 0; i < total_points && sampled < sample_count; i += stride, ++sampled) {
            idx0.clear();
            d20.clear();
            idxn.clear();
            d2n.clear();

            tree.radiusSearch(cloud->points[i], args.tolerance, idx0, d20, 0);
            tree.radiusSearch(cloud->points[i], args.tolerance, idxn, d2n, args.max_n);

            const int c0 = static_cast<int>(idx0.size());
            const int cn = static_cast<int>(idxn.size());
            sum0 += c0;
            sumn += cn;
            max0 = std::max(max0, c0);
            maxn = std::max(maxn, cn);
            if (c0 > args.max_n) ++over_limit_without_cap;
            if (cn == args.max_n && c0 > args.max_n) ++capped_count;

            if (sampled < args.preview) {
                std::cout << "  point[" << i << "] "
                          << "neighbors(max_n=0)=" << c0
                          << " neighbors(max_n=" << args.max_n << ")=" << cn
                          << "\n";
            }
        }

        const double avg0 = sampled > 0 ? static_cast<double>(sum0) / sampled : 0.0;
        const double avgn = sampled > 0 ? static_cast<double>(sumn) / sampled : 0.0;
        const double over_rate = sampled > 0 ? static_cast<double>(over_limit_without_cap) / sampled : 0.0;
        const double cap_rate = sampled > 0 ? static_cast<double>(capped_count) / sampled : 0.0;

        std::cout << std::fixed << std::setprecision(6)
                  << "Sampled points: " << sampled << "\n"
                  << "avg_neighbors(max_n=0)=" << avg0 << "\n"
                  << "avg_neighbors(max_n=" << args.max_n << ")=" << avgn << "\n"
                  << "max_neighbors(max_n=0)=" << max0 << "\n"
                  << "max_neighbors(max_n=" << args.max_n << ")=" << maxn << "\n"
                  << "fraction_with_true_neighbors_over_" << args.max_n << "=" << over_rate << "\n"
                  << "fraction_actually_capped_at_" << args.max_n << "=" << cap_rate << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
