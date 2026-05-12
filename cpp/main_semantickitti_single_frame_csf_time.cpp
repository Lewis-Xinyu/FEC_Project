#pragma warning(disable:4996)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "EC.h"
#include "FEC.h"
#include "FECunion.h"
#include "RG.h"
#include "semantic_kitti_paths.h"
#include "../third_party/csf/src/CSF.h"

namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;

struct Args {
    std::string root;
    std::vector<std::string> sequences;
    std::filesystem::path out_dir = "reports/semantic_kitti_single_frame_csf";
    std::string algorithm = "FECunion";
    int frames_per_sequence = 100;
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    float rg_smoothness_threshold = 3.0f;
    float rg_curvature_threshold = 1.0f;
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

struct FilteredFrame {
    Cloud::Ptr cloud = std::make_shared<Cloud>();
    int raw_points = 0;
    int removed_ground_points = 0;
    double csf_ms = 0.0;
};

struct FrameResult {
    std::string sequence;
    std::string frame_id;
    int raw_points = 0;
    int removed_ground_points = 0;
    int nonground_points = 0;
    double csf_ms = 0.0;
    double cluster_ms = 0.0;
    double total_ms = 0.0;
    double cluster_fps = 0.0;
    double total_fps = 0.0;
    int clusters = 0;
};

struct SequenceSummary {
    std::string sequence;
    int frames = 0;
    double avg_csf_ms = 0.0;
    double avg_cluster_ms = 0.0;
    double avg_total_ms = 0.0;
    double avg_cluster_fps = 0.0;
    double avg_total_fps = 0.0;
    double avg_nonground_points = 0.0;
    int min_nonground_points = 0;
    int max_nonground_points = 0;
};

static std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) parts.push_back(item);
    }
    return parts;
}

static bool parse_bool01(const std::string& value) {
    if (value == "1" || value == "true" || value == "True") return true;
    if (value == "0" || value == "false" || value == "False") return false;
    throw std::runtime_error("Expected 0/1/true/false, got: " + value);
}

static void print_help(const char* argv0) {
    std::cout
        << "SemanticKITTI single-frame CSF timing\n"
        << "Usage:\n"
        << "  " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --root PATH                 SemanticKITTI root\n"
        << "  --sequences LIST            Comma-separated sequence ids, default: all available\n"
        << "  --alg NAME                  FEC | FECunion | EC | RG\n"
        << "  --frames-per-seq N          Frames per sequence, default 100\n"
        << "  --tol FLOAT                 Clustering tolerance, default 0.2\n"
        << "  --min-cluster-size N        Minimum cluster size, default 100\n"
        << "  --max-n N                   Max neighbors parameter, default 50\n"
        << "  --rg-smoothness FLOAT       RG smoothness threshold, default 3.0\n"
        << "  --rg-curvature FLOAT        RG curvature threshold, default 1.0\n"
        << "  --csf-slope-smooth 0|1      CSF slope smoothing, default 1\n"
        << "  --csf-time-step FLOAT       CSF time step, default 0.65\n"
        << "  --csf-class-threshold FLOAT CSF class threshold, default 0.5\n"
        << "  --csf-cloth-resolution FLOAT CSF cloth resolution, default 1.0\n"
        << "  --csf-rigidness N           CSF rigidness, default 3\n"
        << "  --csf-iterations N          CSF iterations, default 500\n"
        << "  --out-dir PATH              CSV output directory\n";
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
        } else if (key == "--sequences") {
            auto v = need(key); if (!v) return false; args.sequences = split_csv(*v);
        } else if (key == "--alg") {
            auto v = need(key); if (!v) return false; args.algorithm = *v;
        } else if (key == "--frames-per-seq") {
            auto v = need(key); if (!v) return false; args.frames_per_sequence = std::stoi(*v);
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else if (key == "--rg-smoothness") {
            auto v = need(key); if (!v) return false; args.rg_smoothness_threshold = std::stof(*v);
        } else if (key == "--rg-curvature") {
            auto v = need(key); if (!v) return false; args.rg_curvature_threshold = std::stof(*v);
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
        } else if (key == "--out-dir") {
            auto v = need(key); if (!v) return false; args.out_dir = *v;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            print_help(argv[0]);
            return false;
        }
    }
    return true;
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

static std::vector<std::string> list_sequences(const std::filesystem::path& sequences_root) {
    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(sequences_root)) {
        if (!entry.is_directory()) continue;
        const auto seq = entry.path().filename().string();
        if (std::filesystem::exists(entry.path() / "velodyne")) ids.push_back(seq);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static std::vector<std::string> list_frame_ids(const std::filesystem::path& velodyne_dir) {
    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(velodyne_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".bin") continue;
        ids.push_back(entry.path().stem().string());
    }
    std::sort(ids.begin(), ids.end());
    return ids;
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

static FilteredFrame load_filtered_frame_with_csf(
    const std::filesystem::path& bin_path,
    const Args& args
) {
    const auto raw = load_bin(bin_path);

    std::vector<csf::Point> points;
    points.reserve(raw.size());
    for (const auto& p : raw) {
        points.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
    }

    FilteredFrame filtered;
    filtered.raw_points = static_cast<int>(raw.size());

    auto csf = make_csf(args);
    const auto t0 = std::chrono::steady_clock::now();
    csf.setPointCloud(points);
    std::vector<int> ground_indexes;
    std::vector<int> offground_indexes;
    csf.do_filtering(ground_indexes, offground_indexes, false);
    const auto t1 = std::chrono::steady_clock::now();

    filtered.csf_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    filtered.removed_ground_points = static_cast<int>(ground_indexes.size());
    filtered.cloud->reserve(offground_indexes.size());
    for (const int idx : offground_indexes) {
        const auto& p = raw[static_cast<std::size_t>(idx)];
        filtered.cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    filtered.cloud->width = static_cast<std::uint32_t>(filtered.cloud->size());
    filtered.cloud->height = 1;
    filtered.cloud->is_dense = false;
    return filtered;
}

struct ScopedSilence {
    std::streambuf* cout_buf = nullptr;
    std::streambuf* cerr_buf = nullptr;
    std::ofstream null;

    ScopedSilence() : null("/dev/null") {
        cout_buf = std::cout.rdbuf(null.rdbuf());
        cerr_buf = std::cerr.rdbuf(null.rdbuf());
    }

    ~ScopedSilence() {
        std::cout.rdbuf(cout_buf);
        std::cerr.rdbuf(cerr_buf);
    }
};

static std::vector<pcl::PointIndices> run_algorithm(const Cloud::Ptr& cloud, const Args& args) {
    ScopedSilence silence;
    if (args.algorithm == "FEC") return FEC(cloud, args.min_cluster_size, args.tolerance, args.max_n);
    if (args.algorithm == "FECunion") return FECunion(cloud, args.min_cluster_size, args.tolerance, args.max_n);
    if (args.algorithm == "EC") return EC(cloud, args.tolerance, args.min_cluster_size);
    if (args.algorithm == "RG") {
        return RG(cloud, args.min_cluster_size, args.max_n, args.rg_smoothness_threshold, args.rg_curvature_threshold);
    }
    throw std::runtime_error("Unknown algorithm: " + args.algorithm);
}

static double fps_from_ms(double ms) {
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

static void write_frame_csv(const Args& args, const std::vector<FrameResult>& rows) {
    const auto path = args.out_dir / ("single_frame_csf_" + args.algorithm + "_frames.csv");
    std::ofstream out(path);
    out << "sequence,frame_id,raw_points,removed_ground_points,nonground_points,algorithm,csf_ms,cluster_ms,total_ms,cluster_fps,total_fps,clusters\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.sequence << ','
            << row.frame_id << ','
            << row.raw_points << ','
            << row.removed_ground_points << ','
            << row.nonground_points << ','
            << args.algorithm << ','
            << row.csf_ms << ','
            << row.cluster_ms << ','
            << row.total_ms << ','
            << row.cluster_fps << ','
            << row.total_fps << ','
            << row.clusters << '\n';
    }
}

static void write_summary_csv(const Args& args, const std::vector<SequenceSummary>& rows) {
    const auto path = args.out_dir / ("single_frame_csf_" + args.algorithm + "_summary.csv");
    std::ofstream out(path);
    out << "sequence,frames,algorithm,avg_csf_ms,avg_cluster_ms,avg_total_ms,avg_cluster_fps,avg_total_fps,avg_nonground_points,min_nonground_points,max_nonground_points\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.sequence << ','
            << row.frames << ','
            << args.algorithm << ','
            << row.avg_csf_ms << ','
            << row.avg_cluster_ms << ','
            << row.avg_total_ms << ','
            << row.avg_cluster_fps << ','
            << row.avg_total_fps << ','
            << row.avg_nonground_points << ','
            << row.min_nonground_points << ','
            << row.max_nonground_points << '\n';
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
        if (args.sequences.empty()) args.sequences = list_sequences(sequences_root);
        std::filesystem::create_directories(args.out_dir);

        std::vector<FrameResult> frame_rows;
        std::vector<SequenceSummary> sequence_rows;

        std::cout << "Single-frame CSF timing\n";
        std::cout << "Algorithm: " << args.algorithm << "\n";
        std::cout << "CSF params: resolution=" << args.csf_cloth_resolution
                  << " threshold=" << args.csf_class_threshold
                  << " rigidness=" << args.csf_rigidness
                  << " iterations=" << args.csf_iterations
                  << " slope_smooth=" << (args.csf_slope_smooth ? 1 : 0)
                  << "\n";
        std::cout << "Sequences:";
        for (const auto& seq : args.sequences) std::cout << " " << seq;
        std::cout << "\n";

        for (const auto& sequence : args.sequences) {
            const auto sequence_root = sequences_root / sequence;
            const auto frame_ids = list_frame_ids(sequence_root / "velodyne");
            const int frames_to_run = std::min<int>(args.frames_per_sequence, static_cast<int>(frame_ids.size()));
            if (frames_to_run <= 0) continue;

            std::vector<double> csf_ms_values;
            std::vector<double> cluster_ms_values;
            std::vector<double> total_ms_values;
            std::vector<int> nonground_counts;
            csf_ms_values.reserve(frames_to_run);
            cluster_ms_values.reserve(frames_to_run);
            total_ms_values.reserve(frames_to_run);
            nonground_counts.reserve(frames_to_run);

            std::cout << "Sequence " << sequence << " frames=" << frames_to_run << "\n";
            for (int i = 0; i < frames_to_run; ++i) {
                const auto& frame_id = frame_ids[i];
                const auto filtered = load_filtered_frame_with_csf(
                    sequence_root / "velodyne" / (frame_id + ".bin"),
                    args
                );

                const auto t0 = std::chrono::steady_clock::now();
                const auto clusters = run_algorithm(filtered.cloud, args);
                const auto t1 = std::chrono::steady_clock::now();
                const double cluster_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                const double total_ms = filtered.csf_ms + cluster_ms;

                FrameResult row;
                row.sequence = sequence;
                row.frame_id = frame_id;
                row.raw_points = filtered.raw_points;
                row.removed_ground_points = filtered.removed_ground_points;
                row.nonground_points = static_cast<int>(filtered.cloud->size());
                row.csf_ms = filtered.csf_ms;
                row.cluster_ms = cluster_ms;
                row.total_ms = total_ms;
                row.cluster_fps = fps_from_ms(cluster_ms);
                row.total_fps = fps_from_ms(total_ms);
                row.clusters = static_cast<int>(clusters.size());
                frame_rows.push_back(row);

                csf_ms_values.push_back(row.csf_ms);
                cluster_ms_values.push_back(row.cluster_ms);
                total_ms_values.push_back(row.total_ms);
                nonground_counts.push_back(row.nonground_points);
            }

            const double avg_csf_ms = std::accumulate(csf_ms_values.begin(), csf_ms_values.end(), 0.0) / static_cast<double>(csf_ms_values.size());
            const double avg_cluster_ms = std::accumulate(cluster_ms_values.begin(), cluster_ms_values.end(), 0.0) / static_cast<double>(cluster_ms_values.size());
            const double avg_total_ms = std::accumulate(total_ms_values.begin(), total_ms_values.end(), 0.0) / static_cast<double>(total_ms_values.size());
            const double avg_points = std::accumulate(nonground_counts.begin(), nonground_counts.end(), 0.0) / static_cast<double>(nonground_counts.size());

            SequenceSummary summary;
            summary.sequence = sequence;
            summary.frames = frames_to_run;
            summary.avg_csf_ms = avg_csf_ms;
            summary.avg_cluster_ms = avg_cluster_ms;
            summary.avg_total_ms = avg_total_ms;
            summary.avg_cluster_fps = fps_from_ms(avg_cluster_ms);
            summary.avg_total_fps = fps_from_ms(avg_total_ms);
            summary.avg_nonground_points = avg_points;
            summary.min_nonground_points = *std::min_element(nonground_counts.begin(), nonground_counts.end());
            summary.max_nonground_points = *std::max_element(nonground_counts.begin(), nonground_counts.end());
            sequence_rows.push_back(summary);

            std::cout << std::fixed << std::setprecision(3)
                      << "  avg_csf_ms=" << summary.avg_csf_ms
                      << " avg_cluster_ms=" << summary.avg_cluster_ms
                      << " avg_total_ms=" << summary.avg_total_ms
                      << " avg_cluster_fps=" << summary.avg_cluster_fps
                      << " avg_total_fps=" << summary.avg_total_fps
                      << " avg_nonground_points=" << summary.avg_nonground_points
                      << "\n";
        }

        write_frame_csv(args, frame_rows);
        write_summary_csv(args, sequence_rows);

        std::cout << "Frame CSV: " << (args.out_dir / ("single_frame_csf_" + args.algorithm + "_frames.csv")) << "\n";
        std::cout << "Summary CSV: " << (args.out_dir / ("single_frame_csf_" + args.algorithm + "_summary.csv")) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
