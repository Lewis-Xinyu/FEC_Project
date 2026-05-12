#pragma warning(disable:4996)

#include <algorithm>
#include <chrono>
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
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "FEC_benchmark.h"
#include "FECunion_benchmark.h"
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
    std::vector<std::string> sequences;
    std::string algorithm = "FECunion";
    int frames_per_sequence = 0;
    int repeats = 7;
    int trim_each_side = 1;
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    std::filesystem::path out_dir = "reports/semantic_kitti_single_frame_maxn_compare";
};

struct XYZI {
    float x;
    float y;
    float z;
    float intensity;
};

struct FrameResult {
    std::string sequence;
    std::string frame_id;
    int raw_points = 0;
    int removed_ground_points = 0;
    int nonground_points = 0;
    double trimmed_mean_ms = 0.0;
    double fps = 0.0;
    int clusters = 0;
};

struct SequenceSummary {
    std::string sequence;
    int valid_frames = 0;
    int skipped_frames = 0;
    double avg_ms = 0.0;
    double avg_fps = 0.0;
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

static void print_help(const char* argv0) {
    std::cout
        << "SemanticKITTI single-frame max_n compare benchmark\n"
        << "Usage:\n"
        << "  " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --root PATH              SemanticKITTI root\n"
        << "  --sequences LIST         Comma-separated sequence ids, default all labeled\n"
        << "  --alg NAME               FEC | FECunion\n"
        << "  --frames-per-seq N       Frames per sequence, 0 means all, default 0\n"
        << "  --repeats N              Repeats per frame, default 7\n"
        << "  --trim-each-side N       Remove N min and N max runs, default 1\n"
        << "  --tol FLOAT              Clustering tolerance, default 0.2\n"
        << "  --min-cluster-size N     Minimum cluster size, default 100\n"
        << "  --max-n N                Max neighbors parameter\n"
        << "  --out-dir PATH           CSV output directory\n";
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
        } else if (key == "--repeats") {
            auto v = need(key); if (!v) return false; args.repeats = std::stoi(*v);
        } else if (key == "--trim-each-side") {
            auto v = need(key); if (!v) return false; args.trim_each_side = std::stoi(*v);
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else if (key == "--out-dir") {
            auto v = need(key); if (!v) return false; args.out_dir = *v;
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            print_help(argv[0]);
            return false;
        }
    }

    if (args.algorithm != "FEC" && args.algorithm != "FECunion") {
        std::cerr << "Unsupported algorithm: " << args.algorithm << "\n";
        return false;
    }
    if (args.repeats < 3) {
        std::cerr << "--repeats must be at least 3\n";
        return false;
    }
    if (args.trim_each_side < 0 || args.trim_each_side * 2 >= args.repeats) {
        std::cerr << "Invalid trim setting\n";
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

static std::vector<std::string> list_sequences(const std::filesystem::path& sequences_root) {
    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(sequences_root)) {
        if (!entry.is_directory()) continue;
        const auto seq = entry.path().filename().string();
        if (std::filesystem::exists(entry.path() / "velodyne") &&
            std::filesystem::exists(entry.path() / "labels")) {
            ids.push_back(seq);
        }
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

struct FilteredFrame {
    Cloud::Ptr cloud = std::make_shared<Cloud>();
    int raw_points = 0;
    int removed_ground_points = 0;
};

static FilteredFrame load_filtered_frame(
    const std::filesystem::path& bin_path,
    const std::filesystem::path& label_path
) {
    const auto raw = load_bin(bin_path);
    const auto labels = load_labels(label_path);
    if (raw.size() != labels.size()) {
        throw std::runtime_error("Point/label size mismatch for " + bin_path.string());
    }

    FilteredFrame filtered;
    filtered.raw_points = static_cast<int>(raw.size());
    filtered.cloud->reserve(raw.size());

    for (std::size_t i = 0; i < raw.size(); ++i) {
        const auto semantic = static_cast<std::uint16_t>(labels[i] & 0xffffu);
        if (is_ground_semantic(semantic)) {
            ++filtered.removed_ground_points;
            continue;
        }
        filtered.cloud->push_back(pcl::PointXYZ(raw[i].x, raw[i].y, raw[i].z));
    }

    filtered.cloud->width = static_cast<std::uint32_t>(filtered.cloud->size());
    filtered.cloud->height = 1;
    filtered.cloud->is_dense = false;
    return filtered;
}

static std::vector<pcl::PointIndices> run_algorithm(const Cloud::Ptr& cloud, const Args& args) {
    if (args.algorithm == "FEC") {
        return fec_benchmark::run_fec(cloud, args.min_cluster_size, args.tolerance, args.max_n);
    }
    return fec_benchmark::run_fecunion(cloud, args.min_cluster_size, args.tolerance, args.max_n);
}

static double fps_from_ms(double ms) {
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

static double trimmed_mean(std::vector<double> values, int trim_each_side) {
    std::sort(values.begin(), values.end());
    const auto begin = values.begin() + trim_each_side;
    const auto end = values.end() - trim_each_side;
    const double sum = std::accumulate(begin, end, 0.0);
    return sum / static_cast<double>(std::distance(begin, end));
}

static void write_frame_csv(const Args& args, const std::vector<FrameResult>& rows) {
    const auto path = args.out_dir / ("single_frame_" + args.algorithm + "_maxn" + std::to_string(args.max_n) + "_frames.csv");
    std::ofstream out(path);
    out << "sequence,frame_id,raw_points,removed_ground_points,nonground_points,algorithm,max_n,repeats,trim_each_side,trimmed_mean_ms,fps,clusters\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.sequence << ','
            << row.frame_id << ','
            << row.raw_points << ','
            << row.removed_ground_points << ','
            << row.nonground_points << ','
            << args.algorithm << ','
            << args.max_n << ','
            << args.repeats << ','
            << args.trim_each_side << ','
            << row.trimmed_mean_ms << ','
            << row.fps << ','
            << row.clusters << '\n';
    }
}

static void write_summary_csv(const Args& args, const std::vector<SequenceSummary>& rows) {
    const auto path = args.out_dir / ("single_frame_" + args.algorithm + "_maxn" + std::to_string(args.max_n) + "_summary.csv");
    std::ofstream out(path);
    out << "sequence,valid_frames,skipped_frames,algorithm,max_n,repeats,trim_each_side,avg_ms,avg_fps,avg_nonground_points,min_nonground_points,max_nonground_points\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.sequence << ','
            << row.valid_frames << ','
            << row.skipped_frames << ','
            << args.algorithm << ','
            << args.max_n << ','
            << args.repeats << ','
            << args.trim_each_side << ','
            << row.avg_ms << ','
            << row.avg_fps << ','
            << row.avg_nonground_points << ','
            << row.min_nonground_points << ','
            << row.max_nonground_points << '\n';
    }
}

static void write_overall_csv(const Args& args, const std::vector<FrameResult>& rows, int skipped_frames) {
    const auto path = args.out_dir / ("single_frame_" + args.algorithm + "_maxn" + std::to_string(args.max_n) + "_overall.csv");
    std::ofstream out(path);
    out << "algorithm,max_n,repeats,trim_each_side,valid_frames,skipped_frames,avg_ms,avg_fps,avg_nonground_points,min_nonground_points,max_nonground_points\n";
    out << std::fixed << std::setprecision(6);
    if (rows.empty()) return;

    double total_ms = 0.0;
    double total_points = 0.0;
    int min_points = std::numeric_limits<int>::max();
    int max_points = 0;
    for (const auto& row : rows) {
        total_ms += row.trimmed_mean_ms;
        total_points += row.nonground_points;
        min_points = std::min(min_points, row.nonground_points);
        max_points = std::max(max_points, row.nonground_points);
    }
    const double avg_ms = total_ms / static_cast<double>(rows.size());
    out << args.algorithm << ','
        << args.max_n << ','
        << args.repeats << ','
        << args.trim_each_side << ','
        << rows.size() << ','
        << skipped_frames << ','
        << avg_ms << ','
        << fps_from_ms(avg_ms) << ','
        << total_points / static_cast<double>(rows.size()) << ','
        << min_points << ','
        << max_points << '\n';
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
        int overall_skipped = 0;

        std::cout << "Single-frame max_n benchmark\n";
        std::cout << "Algorithm: " << args.algorithm << " max_n=" << args.max_n
                  << " repeats=" << args.repeats << " trim=" << args.trim_each_side << "\n";

        for (const auto& sequence : args.sequences) {
            const auto sequence_root = sequences_root / sequence;
            const auto frame_ids = list_frame_ids(sequence_root / "velodyne");
            const int frames_to_run =
                args.frames_per_sequence <= 0
                    ? static_cast<int>(frame_ids.size())
                    : std::min<int>(args.frames_per_sequence, static_cast<int>(frame_ids.size()));
            if (frames_to_run <= 0) continue;

            std::vector<double> ms_values;
            std::vector<int> nonground_counts;
            int skipped_frames = 0;

            std::cout << "Sequence " << sequence << " frames=" << frames_to_run << "\n";
            for (int frame_idx = 0; frame_idx < frames_to_run; ++frame_idx) {
                const auto& frame_id = frame_ids[frame_idx];
                FilteredFrame filtered;
                try {
                    filtered = load_filtered_frame(
                        sequence_root / "velodyne" / (frame_id + ".bin"),
                        sequence_root / "labels" / (frame_id + ".label")
                    );
                } catch (const std::exception& e) {
                    ++skipped_frames;
                    ++overall_skipped;
                    continue;
                }

                std::vector<double> runs;
                runs.reserve(args.repeats);
                int clusters = 0;
                for (int r = 0; r < args.repeats; ++r) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const auto cluster_indices = run_algorithm(filtered.cloud, args);
                    const auto t1 = std::chrono::steady_clock::now();
                    if (r == 0) clusters = static_cast<int>(cluster_indices.size());
                    runs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                }

                const double mean_ms = trimmed_mean(runs, args.trim_each_side);
                FrameResult row;
                row.sequence = sequence;
                row.frame_id = frame_id;
                row.raw_points = filtered.raw_points;
                row.removed_ground_points = filtered.removed_ground_points;
                row.nonground_points = static_cast<int>(filtered.cloud->size());
                row.trimmed_mean_ms = mean_ms;
                row.fps = fps_from_ms(mean_ms);
                row.clusters = clusters;
                frame_rows.push_back(row);

                ms_values.push_back(mean_ms);
                nonground_counts.push_back(row.nonground_points);
            }

            if (ms_values.empty()) continue;

            SequenceSummary summary;
            summary.sequence = sequence;
            summary.valid_frames = static_cast<int>(ms_values.size());
            summary.skipped_frames = skipped_frames;
            summary.avg_ms = std::accumulate(ms_values.begin(), ms_values.end(), 0.0) / static_cast<double>(ms_values.size());
            summary.avg_fps = fps_from_ms(summary.avg_ms);
            summary.avg_nonground_points =
                std::accumulate(nonground_counts.begin(), nonground_counts.end(), 0.0) / static_cast<double>(nonground_counts.size());
            summary.min_nonground_points = *std::min_element(nonground_counts.begin(), nonground_counts.end());
            summary.max_nonground_points = *std::max_element(nonground_counts.begin(), nonground_counts.end());
            sequence_rows.push_back(summary);

            std::cout << std::fixed << std::setprecision(3)
                      << "  avg_ms=" << summary.avg_ms
                      << " avg_fps=" << summary.avg_fps
                      << " valid=" << summary.valid_frames
                      << " skipped=" << summary.skipped_frames
                      << " avg_points=" << summary.avg_nonground_points
                      << "\n";
        }

        write_frame_csv(args, frame_rows);
        write_summary_csv(args, sequence_rows);
        write_overall_csv(args, frame_rows, overall_skipped);

        std::cout << "Frame CSV: " << (args.out_dir / ("single_frame_" + args.algorithm + "_maxn" + std::to_string(args.max_n) + "_frames.csv")) << "\n";
        std::cout << "Summary CSV: " << (args.out_dir / ("single_frame_" + args.algorithm + "_maxn" + std::to_string(args.max_n) + "_summary.csv")) << "\n";
        std::cout << "Overall CSV: " << (args.out_dir / ("single_frame_" + args.algorithm + "_maxn" + std::to_string(args.max_n) + "_overall.csv")) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
