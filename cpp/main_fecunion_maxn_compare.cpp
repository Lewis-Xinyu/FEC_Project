#pragma warning(disable:4996)

#include <algorithm>
#include <cctype>
#include <cstdint>
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

#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "FECunion_profile.h"
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
    int frames_per_sequence = 200;
    std::filesystem::path fused_dir = "reports/semantic_kitti_seq00_static";
    std::string fusion_sequence = "00";
    std::vector<int> targets = {50000, 200000, 500000, 1000000, 1500000, 2000000, 3000000};
    double tolerance = 0.2;
    int min_cluster_size = 100;
    std::vector<int> max_n_values = {0, 50};
    std::filesystem::path out_dir = "reports/maxn_profile/expanded_tol02";
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

struct MetricAccum {
    int items = 0;
    double total_ms = 0.0;
    double total_search_ms = 0.0;
    std::uint64_t total_search_calls = 0;
};

struct CompareRow {
    std::string sample;
    std::string mode;
    std::string sequence;
    int frames_used = 0;
    int target_points = 0;
    double total_ms_0 = 0.0;
    double total_ms_50 = 0.0;
    double avg_search_ms_0 = 0.0;
    double avg_search_ms_50 = 0.0;
    double avg_search_calls_0 = 0.0;
    double avg_search_calls_50 = 0.0;
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
        << "FECunion max_n compare\n"
        << "Usage:\n"
        << "  " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --root PATH                SemanticKITTI root\n"
        << "  --sequences LIST           Comma-separated ids, default all labeled sequences\n"
        << "  --frames-per-seq N         Frames per sequence for single-frame average, default 200\n"
        << "  --fused-dir PATH           Fusion build directory\n"
        << "  --fusion-sequence ID       Fusion sequence id for manifest, default 00\n"
        << "  --targets LIST             Target points list, default 50000,200000,500000,1000000,1500000,2000000,3000000\n"
        << "  --tol FLOAT                Clustering tolerance, default 0.2\n"
        << "  --min-cluster-size N       Minimum cluster size, default 100\n"
        << "  --max-n-values LIST        Compare values, default 0,50\n"
        << "  --out-dir PATH             Output directory\n";
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
        } else if (key == "--frames-per-seq") {
            auto v = need(key); if (!v) return false; args.frames_per_sequence = std::stoi(*v);
        } else if (key == "--fused-dir") {
            auto v = need(key); if (!v) return false; args.fused_dir = *v;
        } else if (key == "--fusion-sequence") {
            auto v = need(key); if (!v) return false; args.fusion_sequence = *v;
        } else if (key == "--targets") {
            auto v = need(key); if (!v) return false; args.targets = parse_int_csv(*v);
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n-values") {
            auto v = need(key); if (!v) return false; args.max_n_values = parse_int_csv(*v);
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

static Cloud::Ptr load_filtered_frame(
    const std::filesystem::path& bin_path,
    const std::filesystem::path& label_path
) {
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

static std::unordered_map<int, std::filesystem::path> load_target_paths(const Args& args) {
    const auto manifest = args.fused_dir / ("seq" + args.fusion_sequence + "_static_fusion_targets.csv");
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("Failed to open manifest: " + manifest.string());

    std::string line;
    std::getline(in, line);
    std::unordered_map<int, std::filesystem::path> out;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < 9) continue;
        out[std::stoi(fields[0])] = resolve_path(fields[8], args.fused_dir);
    }
    return out;
}

static Cloud::Ptr load_ply_cloud(const std::filesystem::path& path) {
    auto cloud = Cloud::Ptr(new Cloud());
    std::ostringstream captured;
    std::streambuf* old_err = std::cerr.rdbuf(captured.rdbuf());
    const int rc = pcl::io::loadPLYFile<pcl::PointXYZ>(path.string(), *cloud);
    std::cerr.rdbuf(old_err);
    if (rc != 0) throw std::runtime_error("Failed to load PLY: " + path.string());
    return cloud;
}

static double avg_search_ms(const MetricAccum& m) {
    return m.total_search_calls > 0
        ? m.total_search_ms / static_cast<double>(m.total_search_calls)
        : 0.0;
}

static double avg_total_ms(const MetricAccum& m) {
    return m.items > 0 ? m.total_ms / static_cast<double>(m.items) : 0.0;
}

static double avg_search_calls(const MetricAccum& m) {
    return m.items > 0 ? static_cast<double>(m.total_search_calls) / static_cast<double>(m.items) : 0.0;
}

static void write_compare_en(
    const std::filesystem::path& path,
    const std::vector<CompareRow>& rows
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write: " + path.string());
    out << "sample,mode,sequence,frames_used,target_points,total_ms_0,total_ms_50,delta_total_ms_50_minus_0,avg_search_ms_0,avg_search_ms_50,delta_avg_search_ms_50_minus_0,avg_search_calls_0,avg_search_calls_50,delta_search_calls_50_minus_0\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.sample << ','
            << row.mode << ','
            << row.sequence << ','
            << row.frames_used << ','
            << row.target_points << ','
            << row.total_ms_0 << ','
            << row.total_ms_50 << ','
            << (row.total_ms_50 - row.total_ms_0) << ','
            << row.avg_search_ms_0 << ','
            << row.avg_search_ms_50 << ','
            << (row.avg_search_ms_50 - row.avg_search_ms_0) << ','
            << row.avg_search_calls_0 << ','
            << row.avg_search_calls_50 << ','
            << (row.avg_search_calls_50 - row.avg_search_calls_0) << '\n';
    }
}

static void write_group_en(
    const std::filesystem::path& path,
    const std::vector<CompareRow>& rows
) {
    double sf_total0 = 0.0, sf_total50 = 0.0, sf_search0 = 0.0, sf_search50 = 0.0, sf_calls0 = 0.0, sf_calls50 = 0.0;
    double fusion_total0 = 0.0, fusion_total50 = 0.0, fusion_search0 = 0.0, fusion_search50 = 0.0, fusion_calls0 = 0.0, fusion_calls50 = 0.0;
    int sf_count = 0, fusion_count = 0;
    for (const auto& row : rows) {
        if (row.mode == "single-frame") {
            ++sf_count;
            sf_total0 += row.total_ms_0;
            sf_total50 += row.total_ms_50;
            sf_search0 += row.avg_search_ms_0;
            sf_search50 += row.avg_search_ms_50;
            sf_calls0 += row.avg_search_calls_0;
            sf_calls50 += row.avg_search_calls_50;
        } else if (row.mode == "fusion") {
            ++fusion_count;
            fusion_total0 += row.total_ms_0;
            fusion_total50 += row.total_ms_50;
            fusion_search0 += row.avg_search_ms_0;
            fusion_search50 += row.avg_search_ms_50;
            fusion_calls0 += row.avg_search_calls_0;
            fusion_calls50 += row.avg_search_calls_50;
        }
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write: " + path.string());
    out << "group,count,avg_total_ms_0,avg_total_ms_50,avg_delta_total_ms_50_minus_0,avg_search_ms_0,avg_search_ms_50,avg_delta_search_ms_50_minus_0,avg_search_calls_0,avg_search_calls_50,avg_delta_search_calls_50_minus_0\n";
    out << std::fixed << std::setprecision(6);
    if (sf_count > 0) {
        out << "single-frame," << sf_count << ','
            << sf_total0 / sf_count << ','
            << sf_total50 / sf_count << ','
            << (sf_total50 - sf_total0) / sf_count << ','
            << sf_search0 / sf_count << ','
            << sf_search50 / sf_count << ','
            << (sf_search50 - sf_search0) / sf_count << ','
            << sf_calls0 / sf_count << ','
            << sf_calls50 / sf_count << ','
            << (sf_calls50 - sf_calls0) / sf_count << '\n';
    }
    if (fusion_count > 0) {
        out << "fusion," << fusion_count << ','
            << fusion_total0 / fusion_count << ','
            << fusion_total50 / fusion_count << ','
            << (fusion_total50 - fusion_total0) / fusion_count << ','
            << fusion_search0 / fusion_count << ','
            << fusion_search50 / fusion_count << ','
            << (fusion_search50 - fusion_search0) / fusion_count << ','
            << fusion_calls0 / fusion_count << ','
            << fusion_calls50 / fusion_count << ','
            << (fusion_calls50 - fusion_calls0) / fusion_count << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args;
        if (!parse_args(argc, argv, args)) return 1;

        std::filesystem::create_directories(args.out_dir);
        const auto sequences_root = semantic_kitti_paths::resolve_sequences_root(
            args.root.empty() ? std::nullopt : std::optional<std::filesystem::path>(args.root)
        );
        if (args.sequences.empty()) args.sequences = list_sequences(sequences_root);
        const auto target_paths = load_target_paths(args);

        std::vector<CompareRow> rows;

        std::cout << "FECunion max_n compare\n";
        std::cout << "Single-frame sequences:";
        for (const auto& seq : args.sequences) std::cout << " " << seq;
        std::cout << "\n";

        for (const auto& sequence : args.sequences) {
            const auto sequence_root = sequences_root / sequence;
            const auto frame_ids = list_frame_ids(sequence_root / "velodyne");
            const int frames_to_run = std::min<int>(args.frames_per_sequence, static_cast<int>(frame_ids.size()));
            if (frames_to_run <= 0) continue;

            std::unordered_map<int, MetricAccum> accum;
            for (int max_n : args.max_n_values) accum[max_n] = MetricAccum{};

            int valid_frames = 0;
            for (int i = 0; i < frames_to_run; ++i) {
                const auto& frame_id = frame_ids[i];
                Cloud::Ptr cloud;
                try {
                    cloud = load_filtered_frame(
                        sequence_root / "velodyne" / (frame_id + ".bin"),
                        sequence_root / "labels" / (frame_id + ".label")
                    );
                } catch (const std::exception& e) {
                    std::cout << "  skip " << sequence << "/" << frame_id << ": " << e.what() << "\n";
                    continue;
                }

                ++valid_frames;
                for (int max_n : args.max_n_values) {
                    auto run = FECunion_profile(cloud, args.min_cluster_size, args.tolerance, max_n);
                    auto& m = accum[max_n];
                    ++m.items;
                    m.total_ms += run.stats.total_ms;
                    m.total_search_ms += run.stats.search_ms;
                    m.total_search_calls += static_cast<std::uint64_t>(run.stats.search_calls);
                }
            }

            CompareRow row;
            row.sample = "single_frame_seq" + sequence + "_avg" + std::to_string(valid_frames);
            row.mode = "single-frame";
            row.sequence = sequence;
            row.frames_used = valid_frames;
            row.target_points = 0;
            row.total_ms_0 = avg_total_ms(accum.at(0));
            row.total_ms_50 = avg_total_ms(accum.at(50));
            row.avg_search_ms_0 = avg_search_ms(accum.at(0));
            row.avg_search_ms_50 = avg_search_ms(accum.at(50));
            row.avg_search_calls_0 = avg_search_calls(accum.at(0));
            row.avg_search_calls_50 = avg_search_calls(accum.at(50));
            rows.push_back(row);

            std::cout << std::fixed << std::setprecision(3)
                      << "  seq " << sequence
                      << " frames=" << valid_frames
                      << " total_ms(0/50)=" << row.total_ms_0 << "/" << row.total_ms_50
                      << " avg_search_ms(0/50)=" << row.avg_search_ms_0 << "/" << row.avg_search_ms_50
                      << " avg_search_calls(0/50)=" << row.avg_search_calls_0 << "/" << row.avg_search_calls_50
                      << "\n";
        }

        std::cout << "Fusion targets:";
        for (int target : args.targets) std::cout << " " << target;
        std::cout << "\n";

        for (int target : args.targets) {
            auto it = target_paths.find(target);
            if (it == target_paths.end()) throw std::runtime_error("Target not found in manifest: " + std::to_string(target));
            const auto cloud = load_ply_cloud(it->second);

            std::unordered_map<int, FECProfileResult> runs;
            for (int max_n : args.max_n_values) {
                runs.emplace(max_n, FECunion_profile(cloud, args.min_cluster_size, args.tolerance, max_n));
            }

            CompareRow row;
            row.sample = "fusion_seq" + args.fusion_sequence + "_" + std::to_string(target);
            row.mode = "fusion";
            row.sequence = args.fusion_sequence;
            row.frames_used = 0;
            row.target_points = target;
            row.total_ms_0 = runs.at(0).stats.total_ms;
            row.total_ms_50 = runs.at(50).stats.total_ms;
            row.avg_search_ms_0 = runs.at(0).stats.search_calls > 0
                ? runs.at(0).stats.search_ms / static_cast<double>(runs.at(0).stats.search_calls)
                : 0.0;
            row.avg_search_ms_50 = runs.at(50).stats.search_calls > 0
                ? runs.at(50).stats.search_ms / static_cast<double>(runs.at(50).stats.search_calls)
                : 0.0;
            row.avg_search_calls_0 = static_cast<double>(runs.at(0).stats.search_calls);
            row.avg_search_calls_50 = static_cast<double>(runs.at(50).stats.search_calls);
            rows.push_back(row);

            std::cout << std::fixed << std::setprecision(3)
                      << "  target " << target
                      << " total_ms(0/50)=" << row.total_ms_0 << "/" << row.total_ms_50
                      << " avg_search_ms(0/50)=" << row.avg_search_ms_0 << "/" << row.avg_search_ms_50
                      << " search_calls(0/50)=" << row.avg_search_calls_0 << "/" << row.avg_search_calls_50
                      << "\n";
        }

        write_compare_en(args.out_dir / "focus_compare_en.csv", rows);
        write_group_en(args.out_dir / "focus_group_summary_en.csv", rows);

        std::cout << "Wrote: " << (args.out_dir / "focus_compare_en.csv") << "\n";
        std::cout << "Wrote: " << (args.out_dir / "focus_group_summary_en.csv") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
