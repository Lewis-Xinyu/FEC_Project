#pragma warning(disable:4996)
#include <algorithm>
#include <array>
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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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

struct Args {
    std::string speed_dataset = "./data/001.ply";
    std::string car_dataset = "./data/car.ply";
    std::string street_dataset = "./data/street.ply";
    std::string report_dir;
    std::vector<std::string> algorithms = {"FECunion", "EC", "RG", "FEC"};
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int min_gt_points = 30;
    int repeat = 7;
    bool write_report = true;
};

enum class PlyFormat {
    kAscii,
    kBinaryLittleEndian
};

struct PlyProperty {
    std::string type;
    std::string name;
};

struct LoadedPly {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std::vector<float> scalar_instance;
    bool has_scalar_instance = false;
};

struct TimingStats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double fps = 0.0;
};

struct InstanceMetrics {
    int points = 0;
    int gt_instances = 0;
    int pred_clusters = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double pq = 0.0;
    double sq = 0.0;
    double rq = 0.0;
    double rc50 = 0.0;
    double miou = 0.0;
    double mean_time_ms = 0.0;
    double fps = 0.0;
};

struct RunOutput {
    std::vector<pcl::PointIndices> clusters;
    double wall_ms = 0.0;
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

static std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            values.push_back(token);
        }
    }
    return values;
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
    cout << "Local report-style clustering benchmark\n"
         << "Usage:\n"
         << "  ./cpp/build/local_report_eval_run [options]\n\n"
         << "Options:\n"
         << "  --speed-dataset PATH       Single-frame speed benchmark input\n"
         << "  --car-dataset PATH         Car-only instance benchmark input\n"
         << "  --street-dataset PATH      Complex-scene instance benchmark input\n"
         << "  --alg NAME[,NAME...]       Algorithm list\n"
         << "  --tol VALUE                Distance tolerance\n"
         << "  --min-cluster-size N       Minimum predicted cluster size\n"
         << "  --max-n N                  Max neighbors for radius search\n"
         << "  --min-gt-points N          Ignore GT instances smaller than this\n"
         << "  --repeat N                 Repeated timing runs per algorithm\n"
         << "  --write-report 0|1         Write markdown report\n"
         << "  --report-dir PATH          Markdown report directory\n"
         << "  --help                     Show this help\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << endl;
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (key == "--help") {
            print_help();
            return false;
        } else if (key == "--speed-dataset") {
            auto value = need_value(key);
            if (!value) return false;
            args.speed_dataset = *value;
        } else if (key == "--car-dataset") {
            auto value = need_value(key);
            if (!value) return false;
            args.car_dataset = *value;
        } else if (key == "--street-dataset") {
            auto value = need_value(key);
            if (!value) return false;
            args.street_dataset = *value;
        } else if (key == "--alg") {
            auto value = need_value(key);
            if (!value) return false;
            args.algorithms = split_csv(*value);
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
        } else if (key == "--repeat") {
            auto value = need_value(key);
            if (!value) return false;
            args.repeat = std::max(1, std::stoi(*value));
        } else if (key == "--write-report") {
            auto value = need_value(key);
            if (!value) return false;
            args.write_report = parse_bool(*value, args.write_report);
        } else if (key == "--report-dir") {
            auto value = need_value(key);
            if (!value) return false;
            args.report_dir = trim(*value);
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

static std::filesystem::path default_report_dir_from_argv0(const char* argv0) {
    std::filesystem::path exe_path = std::filesystem::weakly_canonical(std::filesystem::absolute(argv0));
    return exe_path.parent_path().parent_path().parent_path() / "reports";
}

static std::filesystem::path resolve_input_path(const std::string& text) {
    std::filesystem::path path(text);
    if (std::filesystem::exists(path)) return path;
    std::filesystem::path fallback = std::filesystem::path("/home/rog/FEC_Project") / text;
    if (std::filesystem::exists(fallback)) return fallback;
    throw std::runtime_error("Input file not found: " + text);
}

static std::size_t property_type_size(const std::string& type) {
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
    if (type == "int" || type == "uint" || type == "float" || type == "int32" || type == "uint32" || type == "float32") return 4;
    if (type == "double" || type == "float64" || type == "int64" || type == "uint64") return 8;
    throw std::runtime_error("Unsupported PLY property type: " + type);
}

template <typename T>
static T read_little(const unsigned char* data) {
    T value{};
    std::memcpy(&value, data, sizeof(T));
    return value;
}

static double read_numeric_value(const unsigned char* data, const std::string& type) {
    if (type == "char" || type == "int8") return static_cast<double>(read_little<std::int8_t>(data));
    if (type == "uchar" || type == "uint8") return static_cast<double>(read_little<std::uint8_t>(data));
    if (type == "short" || type == "int16") return static_cast<double>(read_little<std::int16_t>(data));
    if (type == "ushort" || type == "uint16") return static_cast<double>(read_little<std::uint16_t>(data));
    if (type == "int" || type == "int32") return static_cast<double>(read_little<std::int32_t>(data));
    if (type == "uint" || type == "uint32") return static_cast<double>(read_little<std::uint32_t>(data));
    if (type == "float" || type == "float32") return static_cast<double>(read_little<float>(data));
    if (type == "double" || type == "float64") return read_little<double>(data);
    throw std::runtime_error("Unsupported binary property type: " + type);
}

static LoadedPly load_ply_with_instance(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open PLY: " + path.string());
    }

    std::string line;
    PlyFormat format = PlyFormat::kAscii;
    std::size_t vertex_count = 0;
    std::vector<PlyProperty> props;
    bool in_vertex = false;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line == "end_header") break;
        std::stringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "format") {
            std::string value;
            ss >> value;
            if (value == "ascii") {
                format = PlyFormat::kAscii;
            } else if (value == "binary_little_endian") {
                format = PlyFormat::kBinaryLittleEndian;
            } else {
                throw std::runtime_error("Unsupported PLY format in " + path.string());
            }
        } else if (key == "element") {
            std::string name;
            ss >> name;
            if (name == "vertex") {
                ss >> vertex_count;
                in_vertex = true;
                props.clear();
            } else {
                in_vertex = false;
            }
        } else if (key == "property" && in_vertex) {
            std::string type;
            std::string name;
            ss >> type >> name;
            if (type == "list") {
                throw std::runtime_error("PLY list properties are not supported: " + path.string());
            }
            props.push_back({type, name});
        }
    }

    if (vertex_count == 0) {
        throw std::runtime_error("PLY has no vertex data: " + path.string());
    }

    int x_idx = -1;
    int y_idx = -1;
    int z_idx = -1;
    int scalar_idx = -1;
    for (int i = 0; i < static_cast<int>(props.size()); ++i) {
        if (props[i].name == "x") x_idx = i;
        if (props[i].name == "y") y_idx = i;
        if (props[i].name == "z") z_idx = i;
        if (props[i].name == "scalar_instance") scalar_idx = i;
    }
    if (x_idx < 0 || y_idx < 0 || z_idx < 0) {
        throw std::runtime_error("PLY missing x/y/z properties: " + path.string());
    }

    LoadedPly loaded;
    loaded.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    loaded.cloud->reserve(vertex_count);
    loaded.has_scalar_instance = scalar_idx >= 0;
    if (loaded.has_scalar_instance) {
        loaded.scalar_instance.reserve(vertex_count);
    }

    if (format == PlyFormat::kAscii) {
        for (std::size_t i = 0; i < vertex_count; ++i) {
            if (!std::getline(in, line)) {
                throw std::runtime_error("Unexpected EOF in ASCII PLY: " + path.string());
            }
            std::stringstream ss(line);
            std::vector<double> values;
            values.reserve(props.size());
            for (std::size_t p = 0; p < props.size(); ++p) {
                double value = 0.0;
                ss >> value;
                values.push_back(value);
            }
            loaded.cloud->push_back(pcl::PointXYZ(
                static_cast<float>(values[x_idx]),
                static_cast<float>(values[y_idx]),
                static_cast<float>(values[z_idx])
            ));
            if (loaded.has_scalar_instance) {
                loaded.scalar_instance.push_back(static_cast<float>(values[scalar_idx]));
            }
        }
    } else {
        std::vector<std::size_t> offsets(props.size(), 0);
        std::size_t stride = 0;
        for (int i = 0; i < static_cast<int>(props.size()); ++i) {
            offsets[i] = stride;
            stride += property_type_size(props[i].type);
        }
        std::vector<unsigned char> row(stride);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
            if (!in) {
                throw std::runtime_error("Unexpected EOF in binary PLY: " + path.string());
            }
            loaded.cloud->push_back(pcl::PointXYZ(
                static_cast<float>(read_numeric_value(row.data() + offsets[x_idx], props[x_idx].type)),
                static_cast<float>(read_numeric_value(row.data() + offsets[y_idx], props[y_idx].type)),
                static_cast<float>(read_numeric_value(row.data() + offsets[z_idx], props[z_idx].type))
            ));
            if (loaded.has_scalar_instance) {
                loaded.scalar_instance.push_back(static_cast<float>(
                    read_numeric_value(row.data() + offsets[scalar_idx], props[scalar_idx].type)
                ));
            }
        }
    }

    loaded.cloud->width = static_cast<std::uint32_t>(loaded.cloud->size());
    loaded.cloud->height = 1;
    loaded.cloud->is_dense = false;
    return loaded;
}

static RunOutput run_once(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& original_cloud,
    const std::string& algorithm,
    int min_cluster_size,
    double tolerance,
    int max_n
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>(*original_cloud));
    std::ostringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());
    auto t0 = std::chrono::steady_clock::now();
    auto clusters = run_algorithm(cloud, algorithm, min_cluster_size, tolerance, max_n);
    auto t1 = std::chrono::steady_clock::now();
    std::cout.rdbuf(old_buf);

    RunOutput out;
    out.clusters = std::move(clusters);
    out.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

static TimingStats summarize_times(const std::vector<double>& times_ms) {
    TimingStats stats;
    if (times_ms.empty()) return stats;

    stats.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_ms = *std::max_element(times_ms.begin(), times_ms.end());
    stats.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / static_cast<double>(times_ms.size());

    double sq_sum = 0.0;
    for (double value : times_ms) {
        const double diff = value - stats.mean_ms;
        sq_sum += diff * diff;
    }
    stats.std_ms = std::sqrt(sq_sum / static_cast<double>(times_ms.size()));
    stats.fps = stats.mean_ms > 1e-12 ? 1000.0 / stats.mean_ms : 0.0;
    return stats;
}

static TimingStats benchmark_speed_dataset(
    const LoadedPly& data,
    const std::string& algorithm,
    int repeat,
    int min_cluster_size,
    double tolerance,
    int max_n
) {
    std::vector<double> times_ms;
    times_ms.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        auto run = run_once(data.cloud, algorithm, min_cluster_size, tolerance, max_n);
        times_ms.push_back(run.wall_ms);
    }
    return summarize_times(times_ms);
}

static InstanceMetrics evaluate_instance_dataset(
    const LoadedPly& data,
    const std::string& algorithm,
    int repeat,
    int min_cluster_size,
    double tolerance,
    int max_n,
    int min_gt_points
) {
    if (!data.has_scalar_instance) {
        throw std::runtime_error("Dataset has no scalar_instance field.");
    }

    std::unordered_map<int, int> instance_to_index;
    std::vector<std::vector<int>> gt_points;
    gt_points.reserve(64);
    for (int point_idx = 0; point_idx < static_cast<int>(data.scalar_instance.size()); ++point_idx) {
        const int instance_id = static_cast<int>(std::lround(data.scalar_instance[point_idx]));
        if (instance_id <= 0) continue;
        auto it = instance_to_index.find(instance_id);
        if (it == instance_to_index.end()) {
            const int new_index = static_cast<int>(gt_points.size());
            instance_to_index.emplace(instance_id, new_index);
            gt_points.push_back({});
            it = instance_to_index.find(instance_id);
        }
        gt_points[it->second].push_back(point_idx);
    }

    std::vector<std::vector<int>> filtered_gt;
    filtered_gt.reserve(gt_points.size());
    std::vector<int> point_to_gt(data.cloud->size(), -1);
    for (const auto& indices : gt_points) {
        if (static_cast<int>(indices.size()) < min_gt_points) continue;
        const int gt_idx = static_cast<int>(filtered_gt.size());
        filtered_gt.push_back(indices);
        for (int point_idx : indices) {
            point_to_gt[point_idx] = gt_idx;
        }
    }

    std::vector<double> times_ms;
    times_ms.reserve(repeat);
    RunOutput first_run;
    for (int i = 0; i < repeat; ++i) {
        RunOutput run = run_once(data.cloud, algorithm, min_cluster_size, tolerance, max_n);
        times_ms.push_back(run.wall_ms);
        if (i == 0) {
            first_run = std::move(run);
        }
    }

    std::vector<double> best_iou(filtered_gt.size(), 0.0);
    std::vector<std::tuple<double, int, int>> candidate_matches;

    for (int pred_idx = 0; pred_idx < static_cast<int>(first_run.clusters.size()); ++pred_idx) {
        const auto& cluster = first_run.clusters[pred_idx];
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
            const int gt_size = static_cast<int>(filtered_gt[gt_idx].size());
            const int pred_size = static_cast<int>(cluster.indices.size());
            const double union_size = static_cast<double>(gt_size + pred_size - inter);
            const double iou = union_size > 0.0 ? static_cast<double>(inter) / union_size : 0.0;
            best_iou[gt_idx] = std::max(best_iou[gt_idx], iou);
            if (iou >= 0.5) {
                candidate_matches.emplace_back(iou, gt_idx, pred_idx);
            }
        }
    }

    std::sort(
        candidate_matches.begin(),
        candidate_matches.end(),
        [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); }
    );

    std::vector<bool> gt_used(filtered_gt.size(), false);
    std::vector<bool> pred_used(first_run.clusters.size(), false);
    double matched_iou_sum = 0.0;
    int tp = 0;
    for (const auto& [iou, gt_idx, pred_idx] : candidate_matches) {
        if (gt_used[gt_idx] || pred_used[pred_idx]) continue;
        gt_used[gt_idx] = true;
        pred_used[pred_idx] = true;
        matched_iou_sum += iou;
        ++tp;
    }

    InstanceMetrics metrics;
    metrics.points = static_cast<int>(data.cloud->size());
    metrics.gt_instances = static_cast<int>(filtered_gt.size());
    metrics.pred_clusters = static_cast<int>(first_run.clusters.size());
    metrics.tp = tp;
    metrics.fp = metrics.pred_clusters - tp;
    metrics.fn = metrics.gt_instances - tp;
    metrics.sq = tp > 0 ? matched_iou_sum / static_cast<double>(tp) : 0.0;
    metrics.rq = (tp + metrics.fp + metrics.fn) > 0
        ? static_cast<double>(tp) / (static_cast<double>(tp) + 0.5 * metrics.fp + 0.5 * metrics.fn)
        : 0.0;
    metrics.pq = metrics.sq * metrics.rq;
    metrics.rc50 = (tp + metrics.fn) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + metrics.fn) : 0.0;
    metrics.miou = best_iou.empty()
        ? 0.0
        : std::accumulate(best_iou.begin(), best_iou.end(), 0.0) / static_cast<double>(best_iou.size());
    TimingStats timing = summarize_times(times_ms);
    metrics.mean_time_ms = timing.mean_ms;
    metrics.fps = timing.fps;
    return metrics;
}

static std::string format_double(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

static std::string make_report_body(
    const Args& args,
    const std::filesystem::path& speed_dataset_path,
    const LoadedPly& speed_data,
    const std::unordered_map<std::string, TimingStats>& speed_stats,
    const std::filesystem::path& car_dataset_path,
    const LoadedPly& car_data,
    const std::unordered_map<std::string, InstanceMetrics>& car_stats,
    const std::filesystem::path& street_dataset_path,
    const LoadedPly& street_data,
    const std::unordered_map<std::string, InstanceMetrics>& street_stats
) {
    std::ostringstream out;
    out << "# 本地报告风格实验结果\n\n";
    out << "## 设置说明\n\n";
    out << "- 算法：";
    for (std::size_t i = 0; i < args.algorithms.size(); ++i) {
        if (i) out << "、";
        out << "`" << args.algorithms[i] << "`";
    }
    out << "\n";
    out << "- 公共参数：`tol=" << args.tolerance
        << "`，`min_cluster_size=" << args.min_cluster_size
        << "`，`max_n=" << args.max_n
        << "`，`min_gt_points=" << args.min_gt_points
        << "`，`repeat=" << args.repeat << "`\n";
    out << "- 说明：旧报告依赖的外部数据与原始结果文件当前不在本机，因此这里按同类测法在仓库自带本地数据上重新测量。\n\n";

    out << "## 1. 单帧固定参数纯测速（整帧送入聚类）\n\n";
    out << "- 数据文件：`" << speed_dataset_path.string() << "`\n";
    out << "- 点云数量：`" << speed_data.cloud->size() << "`\n\n";
    out << "| 算法 | Mean(ms) | Std(ms) | Min(ms) | Max(ms) | FPS |\n";
    out << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto& algorithm : args.algorithms) {
        const auto& stats = speed_stats.at(algorithm);
        out << "|" << algorithm
            << "|" << format_double(stats.mean_ms)
            << "|" << format_double(stats.std_ms)
            << "|" << format_double(stats.min_ms)
            << "|" << format_double(stats.max_ms)
            << "|" << format_double(stats.fps)
            << "|\n";
    }

    out << "\n## 2. 单帧 car 实例精度（只用实例标签做聚类评测）\n\n";
    out << "- 数据文件：`" << car_dataset_path.string() << "`\n";
    out << "- 点云数量：`" << car_data.cloud->size() << "`\n";
    out << "- 可用 GT 实例数：`";
    if (!car_stats.empty()) out << car_stats.begin()->second.gt_instances;
    else out << 0;
    out << "`\n\n";
    out << "| 算法 | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU | GT | Pred |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& algorithm : args.algorithms) {
        const auto& stats = car_stats.at(algorithm);
        out << "|" << algorithm
            << "|" << format_double(stats.mean_time_ms)
            << "|" << format_double(stats.fps)
            << "|" << format_double(stats.pq * 100.0)
            << "|" << format_double(stats.sq * 100.0)
            << "|" << format_double(stats.rq * 100.0)
            << "|" << format_double(stats.rc50 * 100.0)
            << "|" << format_double(stats.miou * 100.0)
            << "|" << stats.gt_instances
            << "|" << stats.pred_clusters
            << "|\n";
    }

    out << "\n## 3. 复杂场景补充实验（street.ply 全部实例）\n\n";
    out << "- 数据文件：`" << street_dataset_path.string() << "`\n";
    out << "- 点云数量：`" << street_data.cloud->size() << "`\n";
    out << "- 可用 GT 实例数：`";
    if (!street_stats.empty()) out << street_stats.begin()->second.gt_instances;
    else out << 0;
    out << "`\n";
    out << "- 说明：该文件只有实例标签，没有语义类别，因此这里评测的是“全部带实例标签的目标”，作为复杂场景趋势补充。\n\n";
    out << "| 算法 | Time(ms) | FPS | PQ | SQ | RQ | RC50 | mIoU | GT | Pred |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& algorithm : args.algorithms) {
        const auto& stats = street_stats.at(algorithm);
        out << "|" << algorithm
            << "|" << format_double(stats.mean_time_ms)
            << "|" << format_double(stats.fps)
            << "|" << format_double(stats.pq * 100.0)
            << "|" << format_double(stats.sq * 100.0)
            << "|" << format_double(stats.rq * 100.0)
            << "|" << format_double(stats.rc50 * 100.0)
            << "|" << format_double(stats.miou * 100.0)
            << "|" << stats.gt_instances
            << "|" << stats.pred_clusters
            << "|\n";
    }

    out << "\n## 指标说明\n\n";
    out << "- `PQ = SQ × RQ`\n";
    out << "- `SQ = 匹配成功实例的 IoU 均值`\n";
    out << "- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`\n";
    out << "- `RC50 = TP / (TP + FN)`，匹配条件为 `IoU >= 0.5`\n";
    out << "- `mIoU` 这里按“每个 GT 实例的最佳匹配 IoU 均值”计算\n";
    out << "- `FPS = 1000 / Time(ms)`\n";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 0;
    }

    try {
        const std::filesystem::path speed_dataset_path = resolve_input_path(args.speed_dataset);
        const std::filesystem::path car_dataset_path = resolve_input_path(args.car_dataset);
        const std::filesystem::path street_dataset_path = resolve_input_path(args.street_dataset);

        LoadedPly speed_data = load_ply_with_instance(speed_dataset_path);
        LoadedPly car_data = load_ply_with_instance(car_dataset_path);
        LoadedPly street_data = load_ply_with_instance(street_dataset_path);

        if (!car_data.has_scalar_instance) {
            throw std::runtime_error("car dataset has no scalar_instance field: " + car_dataset_path.string());
        }
        if (!street_data.has_scalar_instance) {
            throw std::runtime_error("street dataset has no scalar_instance field: " + street_dataset_path.string());
        }

        std::unordered_map<std::string, TimingStats> speed_stats;
        std::unordered_map<std::string, InstanceMetrics> car_stats;
        std::unordered_map<std::string, InstanceMetrics> street_stats;

        cout << "Local report-style benchmark\n";
        cout << "Speed dataset: " << speed_dataset_path.string() << " points=" << speed_data.cloud->size() << "\n";
        cout << "Car dataset: " << car_dataset_path.string() << " points=" << car_data.cloud->size() << "\n";
        cout << "Street dataset: " << street_dataset_path.string() << " points=" << street_data.cloud->size() << "\n";
        cout << "Algorithms:";
        for (const auto& alg : args.algorithms) cout << " " << alg;
        cout << "\n\n";

        for (const auto& algorithm : args.algorithms) {
            cout << "[Speed] " << algorithm << " ...\n";
            speed_stats.emplace(
                algorithm,
                benchmark_speed_dataset(speed_data, algorithm, args.repeat, args.min_cluster_size, args.tolerance, args.max_n)
            );
            const auto& speed = speed_stats.at(algorithm);
            cout << std::fixed << std::setprecision(3)
                 << "  mean=" << speed.mean_ms << " ms"
                 << " std=" << speed.std_ms << " ms"
                 << " min=" << speed.min_ms << " ms"
                 << " max=" << speed.max_ms << " ms"
                 << " fps=" << speed.fps << "\n";

            cout << "[Car] " << algorithm << " ...\n";
            car_stats.emplace(
                algorithm,
                evaluate_instance_dataset(car_data, algorithm, args.repeat, args.min_cluster_size, args.tolerance, args.max_n, args.min_gt_points)
            );
            const auto& car = car_stats.at(algorithm);
            cout << "  PQ=" << car.pq * 100.0
                 << " SQ=" << car.sq * 100.0
                 << " RQ=" << car.rq * 100.0
                 << " RC50=" << car.rc50 * 100.0
                 << " mIoU=" << car.miou * 100.0
                 << " time=" << car.mean_time_ms << " ms"
                 << " fps=" << car.fps << "\n";

            cout << "[Street] " << algorithm << " ...\n";
            street_stats.emplace(
                algorithm,
                evaluate_instance_dataset(street_data, algorithm, args.repeat, args.min_cluster_size, args.tolerance, args.max_n, args.min_gt_points)
            );
            const auto& street = street_stats.at(algorithm);
            cout << "  PQ=" << street.pq * 100.0
                 << " SQ=" << street.sq * 100.0
                 << " RQ=" << street.rq * 100.0
                 << " RC50=" << street.rc50 * 100.0
                 << " mIoU=" << street.miou * 100.0
                 << " time=" << street.mean_time_ms << " ms"
                 << " fps=" << street.fps << "\n\n";
        }

        const std::string report_body = make_report_body(
            args,
            speed_dataset_path, speed_data, speed_stats,
            car_dataset_path, car_data, car_stats,
            street_dataset_path, street_data, street_stats
        );

        if (args.write_report) {
            const std::filesystem::path report_dir = args.report_dir.empty()
                ? default_report_dir_from_argv0(argv[0])
                : std::filesystem::path(args.report_dir);
            std::filesystem::create_directories(report_dir);
            const std::filesystem::path report_path = report_dir / "local_report_style_benchmark.md";
            std::ofstream out(report_path);
            if (!out) {
                throw std::runtime_error("Failed to write report: " + report_path.string());
            }
            out << report_body;
            cout << "Report written: " << report_path.string() << "\n";
        } else {
            cout << report_body << endl;
        }
    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}
