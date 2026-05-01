#pragma warning(disable:4996)
#include <algorithm>
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
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
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

constexpr std::uint16_t kCar = 7;
constexpr std::uint16_t kGround = 22;

struct Args {
    std::string root = "/mnt/d/semanticposs";
    std::string algorithm = "FECunion";
    std::string report_dir;
    bool help = false;
    int candidates = 60;
    int keep_stage1 = 15;
    int keep_stage2 = 5;
    int seed = 20260425;
    int min_gt_points = 30;
    int stage1_frames = 30;
    int stage2_frames = 120;
    int fusion_samples = 8;
    int fusion_window = 10;
    int fusion_stride = 30;
    bool final_full = false;
    double precision_drop = 0.05;
    double tol_min = 0.12;
    double tol_max = 0.35;
    int min_cluster_min = 40;
    int min_cluster_max = 240;
    int max_n_min = 20;
    int max_n_max = 100;
};

struct Params {
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
};

struct Candidate {
    int id = 0;
    Params params;
};

struct Sample {
    std::string name;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std::vector<std::uint32_t> labels;
    int raw_points = 0;
    int used_points = 0;
};

struct EvalResult {
    int samples = 0;
    long long points = 0;
    int gt = 0;
    int pred = 0;
    int tp = 0;
    int fp = 0;
    int fn = 0;
    double matched_iou_sum = 0.0;
    double best_iou_sum = 0.0;
    double time_ms = 0.0;
};

struct CandidateResult {
    Candidate candidate;
    EvalResult metrics;
    bool eligible = false;
};

static std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static std::optional<std::string> need_value(int& i, int argc, char** argv, const std::string& name) {
    if (i + 1 >= argc) {
        cerr << "Missing value for " << name << endl;
        return std::nullopt;
    }
    return std::string(argv[++i]);
}

static void print_help() {
    cout << "SemanticPOSS speed-first parameter tuning\n"
         << "Usage:\n"
         << "  ./cpp/build/semanticposs_tune_eval_run [options]\n\n"
         << "Main options:\n"
         << "  --root PATH              SemanticPOSS root containing dataset/sequences/\n"
         << "  --alg NAME               Algorithm to tune, default FECunion\n"
         << "  --report-dir PATH        Directory for markdown/csv reports\n"
         << "  --candidates N           Random candidates, including the baseline, default 60\n"
         << "  --keep-stage1 N          Candidates kept after small car-frame test, default 15\n"
         << "  --keep-stage2 N          Candidates kept after wider car-frame test, default 5\n"
         << "  --precision-drop VALUE   Allowed PQ drop from stage best before speed ranking, default 0.05\n"
         << "  --seed N                 Random seed\n\n"
         << "Sample budget:\n"
         << "  --stage1-frames N        seq00 car-only frames for quick screen, default 30\n"
         << "  --stage2-frames N        multi-sequence car-only frames, default 120\n"
         << "  --fusion-samples N       final seq00 fused car samples, default 8\n"
         << "  --fusion-window N        Frames per fused sample, default 10\n"
         << "  --fusion-stride N        Start stride between fused samples, default 30\n\n"
         << "  --final-full 0|1         Final Top-K validation uses all SemanticPOSS car frames\n\n"
         << "Parameter ranges:\n"
         << "  --tol-min V --tol-max V\n"
         << "  --min-cluster-min N --min-cluster-max N\n"
         << "  --max-n-min N --max-n-max N\n";
}

static bool parse_bool(const std::string& value, bool fallback) {
    std::string lower;
    lower.reserve(value.size());
    for (char ch : value) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") return true;
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") return false;
    return fallback;
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            print_help();
            args.help = true;
            return true;
        }
        auto v = [&]() -> std::optional<std::string> { return need_value(i, argc, argv, key); };
        if (key == "--root") {
            auto x = v(); if (!x) return false; args.root = *x;
        } else if (key == "--alg") {
            auto x = v(); if (!x) return false; args.algorithm = trim(*x);
        } else if (key == "--report-dir") {
            auto x = v(); if (!x) return false; args.report_dir = trim(*x);
        } else if (key == "--candidates") {
            auto x = v(); if (!x) return false; args.candidates = std::stoi(*x);
        } else if (key == "--keep-stage1") {
            auto x = v(); if (!x) return false; args.keep_stage1 = std::stoi(*x);
        } else if (key == "--keep-stage2") {
            auto x = v(); if (!x) return false; args.keep_stage2 = std::stoi(*x);
        } else if (key == "--seed") {
            auto x = v(); if (!x) return false; args.seed = std::stoi(*x);
        } else if (key == "--min-gt-points") {
            auto x = v(); if (!x) return false; args.min_gt_points = std::stoi(*x);
        } else if (key == "--stage1-frames") {
            auto x = v(); if (!x) return false; args.stage1_frames = std::stoi(*x);
        } else if (key == "--stage2-frames") {
            auto x = v(); if (!x) return false; args.stage2_frames = std::stoi(*x);
        } else if (key == "--fusion-samples") {
            auto x = v(); if (!x) return false; args.fusion_samples = std::stoi(*x);
        } else if (key == "--fusion-window") {
            auto x = v(); if (!x) return false; args.fusion_window = std::stoi(*x);
        } else if (key == "--fusion-stride") {
            auto x = v(); if (!x) return false; args.fusion_stride = std::stoi(*x);
        } else if (key == "--final-full") {
            auto x = v(); if (!x) return false; args.final_full = parse_bool(*x, args.final_full);
        } else if (key == "--precision-drop") {
            auto x = v(); if (!x) return false; args.precision_drop = std::stod(*x);
        } else if (key == "--tol-min") {
            auto x = v(); if (!x) return false; args.tol_min = std::stod(*x);
        } else if (key == "--tol-max") {
            auto x = v(); if (!x) return false; args.tol_max = std::stod(*x);
        } else if (key == "--min-cluster-min") {
            auto x = v(); if (!x) return false; args.min_cluster_min = std::stoi(*x);
        } else if (key == "--min-cluster-max") {
            auto x = v(); if (!x) return false; args.min_cluster_max = std::stoi(*x);
        } else if (key == "--max-n-min") {
            auto x = v(); if (!x) return false; args.max_n_min = std::stoi(*x);
        } else if (key == "--max-n-max") {
            auto x = v(); if (!x) return false; args.max_n_max = std::stoi(*x);
        } else {
            cerr << "Unknown argument: " << key << endl;
            return false;
        }
    }
    if (args.candidates < 1 || args.keep_stage1 < 1 || args.keep_stage2 < 1) {
        cerr << "Candidate and keep counts must be positive" << endl;
        return false;
    }
    if (args.tol_min > args.tol_max || args.min_cluster_min > args.min_cluster_max || args.max_n_min > args.max_n_max) {
        cerr << "Invalid parameter range" << endl;
        return false;
    }
    args.keep_stage1 = std::min(args.keep_stage1, args.candidates);
    args.keep_stage2 = std::min(args.keep_stage2, args.keep_stage1);
    return true;
}

static std::filesystem::path sequences_root(const std::filesystem::path& root) {
    if (std::filesystem::exists(root / "sequences")) return root / "sequences";
    if (std::filesystem::exists(root / "dataset" / "sequences")) return root / "dataset" / "sequences";
    throw std::runtime_error("SemanticPOSS root is incomplete: " + root.string());
}

static std::vector<std::string> list_sequences(const std::filesystem::path& seq_root) {
    std::vector<std::string> seqs;
    for (const auto& entry : std::filesystem::directory_iterator(seq_root)) {
        if (entry.is_directory()) seqs.push_back(entry.path().filename().string());
    }
    std::sort(seqs.begin(), seqs.end());
    return seqs;
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

static std::vector<std::string> evenly_select(const std::vector<std::string>& values, int limit) {
    if (limit <= 0 || static_cast<int>(values.size()) <= limit) return values;
    std::vector<std::string> out;
    out.reserve(limit);
    const double step = static_cast<double>(values.size()) / static_cast<double>(limit);
    for (int i = 0; i < limit; ++i) {
        int idx = static_cast<int>(std::floor(i * step));
        idx = std::min(idx, static_cast<int>(values.size()) - 1);
        out.push_back(values[idx]);
    }
    return out;
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_bin(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open point cloud: " + path.string());
    struct XYZI { float x, y, z, intensity; };
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size % static_cast<std::streamsize>(sizeof(XYZI)) != 0) {
        throw std::runtime_error("Unexpected .bin size: " + path.string());
    }
    std::vector<XYZI> raw(static_cast<std::size_t>(size / sizeof(XYZI)));
    in.read(reinterpret_cast<char*>(raw.data()), size);
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(raw.size());
    for (const auto& p : raw) cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static std::vector<std::uint32_t> load_labels(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open label file: " + path.string());
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
        throw std::runtime_error("Unexpected label size: " + path.string());
    }
    std::vector<std::uint32_t> labels(static_cast<std::size_t>(size / sizeof(std::uint32_t)));
    in.read(reinterpret_cast<char*>(labels.data()), size);
    return labels;
}

static std::vector<Eigen::Matrix4d> load_poses(const std::filesystem::path& path) {
    std::ifstream in(path);
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

static bool contains_semantic(const std::set<std::uint16_t>& semantics, std::uint16_t semantic) {
    return semantics.empty() || semantics.count(semantic) > 0;
}

static Sample make_sample(
    const std::filesystem::path& sequence_root,
    const std::vector<std::string>& frame_ids,
    const std::set<std::uint16_t>& include_semantics,
    bool remove_ground,
    bool transform_with_pose,
    const std::string& name
) {
    Sample sample;
    sample.name = name;
    sample.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<Eigen::Matrix4d> poses;
    if (transform_with_pose) poses = load_poses(sequence_root / "poses.txt");

    for (const auto& frame_id : frame_ids) {
        auto cloud = load_bin(sequence_root / "velodyne" / (frame_id + ".bin"));
        auto labels = load_labels(sequence_root / "labels" / (frame_id + ".label"));
        if (cloud->size() != labels.size()) throw std::runtime_error("Point/label mismatch: " + frame_id);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        const int frame_index = std::stoi(frame_id);
        if (transform_with_pose && frame_index >= 0 && frame_index < static_cast<int>(poses.size())) {
            pose = poses[frame_index];
        }
        sample.raw_points += static_cast<int>(cloud->size());
        for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
            const std::uint16_t semantic = static_cast<std::uint16_t>(labels[i] & 0xffffu);
            if (remove_ground && semantic == kGround) continue;
            if (!contains_semantic(include_semantics, semantic)) continue;
            pcl::PointXYZ p = cloud->points[i];
            if (transform_with_pose) {
                Eigen::Vector4d q(p.x, p.y, p.z, 1.0);
                q = pose * q;
                p.x = static_cast<float>(q.x());
                p.y = static_cast<float>(q.y());
                p.z = static_cast<float>(q.z());
            }
            sample.cloud->push_back(p);
            sample.labels.push_back(labels[i]);
        }
    }
    sample.used_points = static_cast<int>(sample.cloud->size());
    sample.cloud->width = static_cast<std::uint32_t>(sample.cloud->size());
    sample.cloud->height = 1;
    sample.cloud->is_dense = false;
    return sample;
}

static std::vector<pcl::PointIndices> run_algorithm(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const std::string& algorithm,
    const Params& params
) {
    if (algorithm == "FEC") return FEC(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC_Block") return FEC_Block(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC1") return FEC1(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC1_1") return FEC1_1(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FECunion") return FECunion(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "EC") return EC(cloud, params.tolerance, params.min_cluster_size);
    if (algorithm == "EC_Block") return EC_Block(cloud, params.tolerance, params.min_cluster_size);
    if (algorithm == "RG") return RG(cloud, params.min_cluster_size, 30, 3.0f, 1.0f);
    if (algorithm == "FEC_Union") return FEC_Union(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC_Union_Block") return FEC_Union_Block(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC_Union_Block_new") return FEC_Union_Block_new(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC_Union_Block_new2") return FEC_Union_Block_new2(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FECunion_Block") return FECunion_Block(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC_Union_Grid_Block") return FEC_Union_Grid_Block(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "Voxel_FEC1") return Voxel_FEC1(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    if (algorithm == "FEC1_Block") return FEC1_Block(cloud, params.min_cluster_size, params.tolerance, params.max_n);
    throw std::runtime_error("Unknown algorithm: " + algorithm);
}

static EvalResult evaluate_sample(
    const Sample& sample,
    const std::string& algorithm,
    const Params& params,
    const Args& args
) {
    std::unordered_map<std::uint32_t, int> gt_key_to_idx;
    std::vector<std::vector<int>> gt_lists;
    std::vector<int> point_to_gt(sample.cloud->size(), -1);
    for (int i = 0; i < static_cast<int>(sample.labels.size()); ++i) {
        const std::uint16_t semantic = static_cast<std::uint16_t>(sample.labels[i] & 0xffffu);
        const std::uint16_t instance = static_cast<std::uint16_t>(sample.labels[i] >> 16);
        if (semantic == 0 || instance == 0 || semantic != kCar) continue;
        const std::uint32_t key = (static_cast<std::uint32_t>(instance) << 16) | semantic;
        auto [it, inserted] = gt_key_to_idx.emplace(key, static_cast<int>(gt_lists.size()));
        if (inserted) gt_lists.push_back({});
        gt_lists[it->second].push_back(i);
    }

    std::vector<std::vector<int>> filtered_gt;
    for (const auto& points : gt_lists) {
        if (static_cast<int>(points.size()) < args.min_gt_points) continue;
        const int gt_idx = static_cast<int>(filtered_gt.size());
        filtered_gt.push_back(points);
        for (int idx : points) point_to_gt[idx] = gt_idx;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>(*sample.cloud));
    std::ostringstream captured;
    std::streambuf* old = std::cout.rdbuf(captured.rdbuf());
    auto t0 = std::chrono::steady_clock::now();
    auto clusters = run_algorithm(cloud, algorithm, params);
    auto t1 = std::chrono::steady_clock::now();
    std::cout.rdbuf(old);

    std::vector<double> best_iou(filtered_gt.size(), 0.0);
    std::vector<std::tuple<double, int, int>> candidates;
    for (int pred_idx = 0; pred_idx < static_cast<int>(clusters.size()); ++pred_idx) {
        std::unordered_map<int, int> overlap;
        for (int point_idx : clusters[pred_idx].indices) {
            if (point_idx < 0 || point_idx >= static_cast<int>(point_to_gt.size())) continue;
            const int gt = point_to_gt[point_idx];
            if (gt >= 0) ++overlap[gt];
        }
        for (const auto& [gt, inter] : overlap) {
            const int gt_size = static_cast<int>(filtered_gt[gt].size());
            const int pred_size = static_cast<int>(clusters[pred_idx].indices.size());
            const double denom = static_cast<double>(gt_size + pred_size - inter);
            const double iou = denom > 0.0 ? static_cast<double>(inter) / denom : 0.0;
            best_iou[gt] = std::max(best_iou[gt], iou);
            if (iou >= 0.5) candidates.emplace_back(iou, gt, pred_idx);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    std::vector<bool> gt_used(filtered_gt.size(), false);
    std::vector<bool> pred_used(clusters.size(), false);

    EvalResult result;
    result.samples = 1;
    result.points = sample.used_points;
    result.gt = static_cast<int>(filtered_gt.size());
    result.pred = static_cast<int>(clusters.size());
    for (const auto& [iou, gt, pred] : candidates) {
        if (gt_used[gt] || pred_used[pred]) continue;
        gt_used[gt] = true;
        pred_used[pred] = true;
        ++result.tp;
        result.matched_iou_sum += iou;
    }
    result.fp = result.pred - result.tp;
    result.fn = result.gt - result.tp;
    result.best_iou_sum = std::accumulate(best_iou.begin(), best_iou.end(), 0.0);
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

static EvalResult merge_results(const std::vector<EvalResult>& results) {
    EvalResult out;
    for (const auto& r : results) {
        out.samples += r.samples;
        out.points += r.points;
        out.gt += r.gt;
        out.pred += r.pred;
        out.tp += r.tp;
        out.fp += r.fp;
        out.fn += r.fn;
        out.matched_iou_sum += r.matched_iou_sum;
        out.best_iou_sum += r.best_iou_sum;
        out.time_ms += r.time_ms;
    }
    if (!results.empty()) out.time_ms /= static_cast<double>(results.size());
    return out;
}

static double sq(const EvalResult& r) { return r.tp > 0 ? r.matched_iou_sum / r.tp : 0.0; }
static double rq(const EvalResult& r) {
    const double denom = r.tp + 0.5 * r.fp + 0.5 * r.fn;
    return denom > 0.0 ? r.tp / denom : 0.0;
}
static double pq(const EvalResult& r) { return sq(r) * rq(r); }
static double rc50(const EvalResult& r) { return (r.tp + r.fn) > 0 ? static_cast<double>(r.tp) / (r.tp + r.fn) : 0.0; }
static double miou(const EvalResult& r) { return r.gt > 0 ? r.best_iou_sum / r.gt : 0.0; }

static std::vector<Candidate> generate_candidates(const Args& args) {
    std::vector<Candidate> out;
    out.reserve(args.candidates);
    out.push_back({0, Params{0.2, 100, 50}});
    std::mt19937 rng(static_cast<std::uint32_t>(args.seed));
    std::uniform_real_distribution<double> tol_dist(args.tol_min, args.tol_max);
    std::uniform_int_distribution<int> min_cluster_dist(args.min_cluster_min, args.min_cluster_max);
    std::uniform_int_distribution<int> max_n_dist(args.max_n_min, args.max_n_max);
    for (int i = 1; i < args.candidates; ++i) {
        Params p;
        p.tolerance = tol_dist(rng);
        p.min_cluster_size = min_cluster_dist(rng);
        p.max_n = max_n_dist(rng);
        out.push_back({i, p});
    }
    return out;
}

static std::vector<Sample> build_stage1_samples(const std::filesystem::path& seq_root, const Args& args) {
    const auto sequence_root = seq_root / "00";
    auto frames = evenly_select(list_frames(sequence_root), args.stage1_frames);
    std::vector<Sample> samples;
    for (const auto& frame : frames) {
        auto sample = make_sample(sequence_root, {frame}, {kCar}, false, false, "00_" + frame + "_car");
        if (sample.used_points > 0) samples.push_back(std::move(sample));
    }
    return samples;
}

static std::vector<Sample> build_stage2_samples(const std::filesystem::path& seq_root, const Args& args) {
    auto seqs = list_sequences(seq_root);
    std::vector<std::pair<std::string, std::string>> all;
    for (const auto& seq : seqs) {
        for (const auto& frame : list_frames(seq_root / seq)) all.emplace_back(seq, frame);
    }
    if (args.stage2_frames > 0 && static_cast<int>(all.size()) > args.stage2_frames) {
        std::vector<std::pair<std::string, std::string>> selected;
        selected.reserve(args.stage2_frames);
        const double step = static_cast<double>(all.size()) / static_cast<double>(args.stage2_frames);
        for (int i = 0; i < args.stage2_frames; ++i) {
            int idx = static_cast<int>(std::floor(i * step));
            idx = std::min(idx, static_cast<int>(all.size()) - 1);
            selected.push_back(all[idx]);
        }
        all = std::move(selected);
    }

    std::vector<Sample> samples;
    for (const auto& [seq, frame] : all) {
        auto sample = make_sample(seq_root / seq, {frame}, {kCar}, false, false, seq + "_" + frame + "_car");
        if (sample.used_points > 0) samples.push_back(std::move(sample));
    }
    return samples;
}

static std::vector<Sample> build_fusion_samples(const std::filesystem::path& seq_root, const Args& args) {
    std::vector<Sample> samples;
    if (args.fusion_samples <= 0) return samples;
    const auto sequence_root = seq_root / "00";
    auto frames = list_frames(sequence_root);
    for (int i = 0; i < args.fusion_samples; ++i) {
        const int start = i * args.fusion_stride;
        if (start + args.fusion_window > static_cast<int>(frames.size())) break;
        std::vector<std::string> window(frames.begin() + start, frames.begin() + start + args.fusion_window);
        auto sample = make_sample(sequence_root, window, {kCar}, false, true, "00_fusion_" + std::to_string(i));
        if (sample.used_points > 0) samples.push_back(std::move(sample));
    }
    return samples;
}

static EvalResult evaluate_candidate(
    const std::vector<Sample>& samples,
    const std::string& algorithm,
    const Candidate& candidate,
    const Args& args
) {
    std::vector<EvalResult> per_sample;
    per_sample.reserve(samples.size());
    for (const auto& sample : samples) {
        per_sample.push_back(evaluate_sample(sample, algorithm, candidate.params, args));
    }
    return merge_results(per_sample);
}

static std::vector<CandidateResult> run_stage(
    const std::string& name,
    const std::vector<Sample>& samples,
    const std::vector<Candidate>& candidates,
    const Args& args
) {
    cout << name << ": candidates=" << candidates.size() << " samples=" << samples.size() << endl;
    std::vector<CandidateResult> results;
    results.reserve(candidates.size());
    for (const auto& c : candidates) {
        CandidateResult cr;
        cr.candidate = c;
        cr.metrics = evaluate_candidate(samples, args.algorithm, c, args);
        results.push_back(cr);
        cout << "  #" << c.id
             << " tol=" << std::fixed << std::setprecision(3) << c.params.tolerance
             << " min=" << c.params.min_cluster_size
             << " max_n=" << c.params.max_n
             << " PQ=" << std::setprecision(2) << pq(cr.metrics) * 100.0
             << " time=" << cr.metrics.time_ms << " ms" << endl;
    }

    const auto best_it = std::max_element(results.begin(), results.end(),
        [](const CandidateResult& a, const CandidateResult& b) { return pq(a.metrics) < pq(b.metrics); });
    const double best_pq = best_it == results.end() ? 0.0 : pq(best_it->metrics);
    const double threshold = best_pq * (1.0 - args.precision_drop);
    for (auto& r : results) r.eligible = pq(r.metrics) >= threshold;

    std::sort(results.begin(), results.end(), [](const CandidateResult& a, const CandidateResult& b) {
        if (a.eligible != b.eligible) return a.eligible > b.eligible;
        if (a.eligible && b.eligible) {
            if (std::abs(a.metrics.time_ms - b.metrics.time_ms) > 1e-9) return a.metrics.time_ms < b.metrics.time_ms;
            return pq(a.metrics) > pq(b.metrics);
        }
        if (std::abs(pq(a.metrics) - pq(b.metrics)) > 1e-12) return pq(a.metrics) > pq(b.metrics);
        return a.metrics.time_ms < b.metrics.time_ms;
    });
    cout << name << " threshold PQ=" << std::fixed << std::setprecision(2) << threshold * 100.0
         << " best candidate=#" << (results.empty() ? -1 : results.front().candidate.id) << endl;
    return results;
}

static std::vector<Candidate> keep_top(const std::vector<CandidateResult>& results, int k) {
    std::vector<Candidate> out;
    for (int i = 0; i < static_cast<int>(results.size()) && i < k; ++i) out.push_back(results[i].candidate);
    return out;
}

static std::filesystem::path default_report_dir(const char* argv0) {
    std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::absolute(argv0));
    return exe.parent_path().parent_path().parent_path() / "reports";
}

static void write_result_row(std::ostream& out, const CandidateResult& r) {
    out << "|" << r.candidate.id
        << "|" << std::fixed << std::setprecision(4) << r.candidate.params.tolerance
        << "|" << r.candidate.params.min_cluster_size
        << "|" << r.candidate.params.max_n
        << "|" << std::setprecision(2) << pq(r.metrics) * 100.0
        << "|" << sq(r.metrics) * 100.0
        << "|" << rq(r.metrics) * 100.0
        << "|" << rc50(r.metrics) * 100.0
        << "|" << miou(r.metrics) * 100.0
        << "|" << r.metrics.time_ms
        << "|" << r.metrics.samples
        << "|" << r.metrics.points
        << "|" << r.metrics.gt
        << "|" << r.metrics.pred
        << "|" << r.metrics.tp
        << "|" << r.metrics.fp
        << "|" << r.metrics.fn
        << "|" << (r.eligible ? "yes" : "no")
        << "|\n";
}

static void write_csv_row(std::ostream& out, const std::string& stage, const CandidateResult& r) {
    out << stage
        << "," << r.candidate.id
        << "," << std::fixed << std::setprecision(6) << r.candidate.params.tolerance
        << "," << r.candidate.params.min_cluster_size
        << "," << r.candidate.params.max_n
        << "," << pq(r.metrics)
        << "," << sq(r.metrics)
        << "," << rq(r.metrics)
        << "," << rc50(r.metrics)
        << "," << miou(r.metrics)
        << "," << r.metrics.time_ms
        << "," << r.metrics.samples
        << "," << r.metrics.points
        << "," << r.metrics.gt
        << "," << r.metrics.pred
        << "," << r.metrics.tp
        << "," << r.metrics.fp
        << "," << r.metrics.fn
        << "," << (r.eligible ? 1 : 0)
        << "\n";
}

static void write_reports(
    const Args& args,
    const std::filesystem::path& report_dir,
    const std::vector<CandidateResult>& stage1,
    const std::vector<CandidateResult>& stage2,
    const std::vector<CandidateResult>& final_stage
) {
    std::filesystem::create_directories(report_dir);
    const auto md_path = report_dir / "semanticposs_tune_report.md";
    const auto csv_path = report_dir / "semanticposs_tune_results.csv";

    std::ofstream md(md_path);
    if (!md) throw std::runtime_error("Failed to write report: " + md_path.string());

    const CandidateResult* best = final_stage.empty() ? nullptr : &final_stage.front();
    md << "# SemanticPOSS Speed-First Parameter Tuning\n\n";
    md << "## Setup\n\n";
    md << "- Algorithm: `" << args.algorithm << "`\n";
    md << "- Candidate count: `" << args.candidates << "`\n";
    md << "- Keep counts: `" << args.keep_stage1 << " -> " << args.keep_stage2 << "`\n";
    md << "- Precision rule: candidates within `" << args.precision_drop * 100.0
       << "%` PQ drop from stage best are ranked by lower time.\n";
    md << "- Final validation: `" << (args.final_full ? "all SemanticPOSS car frames" : "seq00 fusion car samples") << "`\n";
    md << "- Tuned parameters: `tol`, `min_cluster_size`, `max_n`\n";
    md << "- Ranges: `tol=[" << args.tol_min << ", " << args.tol_max << "]`, `min_cluster_size=["
       << args.min_cluster_min << ", " << args.min_cluster_max << "]`, `max_n=["
       << args.max_n_min << ", " << args.max_n_max << "]`\n\n";

    if (best) {
        md << "## Recommended Parameters\n\n";
        md << "| Candidate | tol | min_cluster_size | max_n | PQ | RC50 | mIoU | Time(ms) |\n";
        md << "|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        md << "|" << best->candidate.id
           << "|" << std::fixed << std::setprecision(4) << best->candidate.params.tolerance
           << "|" << best->candidate.params.min_cluster_size
           << "|" << best->candidate.params.max_n
           << "|" << std::setprecision(2) << pq(best->metrics) * 100.0
           << "|" << rc50(best->metrics) * 100.0
           << "|" << miou(best->metrics) * 100.0
           << "|" << best->metrics.time_ms
           << "|\n\n";
    }

    auto table = [&](const std::string& title, const std::vector<CandidateResult>& rows) {
        md << "## " << title << "\n\n";
        md << "| Candidate | tol | min_cluster_size | max_n | PQ | SQ | RQ | RC50 | mIoU | Time(ms) | Samples | Points | GT | Pred | TP | FP | FN | Eligible |\n";
        md << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n";
        for (const auto& r : rows) write_result_row(md, r);
        md << "\n";
    };
    table("Stage 1: Small Car-Frame Screen", stage1);
    table("Stage 2: Multi-Sequence Car-Frame Screen", stage2);
    table(args.final_full ? "Final Stage: Full-Dataset Car Validation" : "Final Stage: Fusion-Car Validation", final_stage);

    md << "## Metric Notes\n\n";
    md << "- `PQ = SQ * RQ`.\n";
    md << "- `SQ` is the mean IoU of one-to-one matches with `IoU >= 0.5`.\n";
    md << "- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.\n";
    md << "- `RC50 = TP / (TP + FN)`.\n";
    md << "- The speed-first choice is the fastest candidate inside the allowed PQ-drop band.\n";

    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("Failed to write CSV: " + csv_path.string());
    csv << "stage,candidate,tol,min_cluster_size,max_n,pq,sq,rq,rc50,miou,time_ms,samples,points,gt,pred,tp,fp,fn,eligible\n";
    for (const auto& r : stage1) write_csv_row(csv, "stage1", r);
    for (const auto& r : stage2) write_csv_row(csv, "stage2", r);
    for (const auto& r : final_stage) write_csv_row(csv, "final", r);

    cout << "Reports written:\n  " << md_path.string() << "\n  " << csv_path.string() << endl;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;
    if (args.help) return 0;

    try {
        const auto seq_root = sequences_root(args.root);
        cout << "Loading stage samples..." << endl;
        auto stage1_samples = build_stage1_samples(seq_root, args);
        auto stage2_samples = build_stage2_samples(seq_root, args);
        if (stage1_samples.empty() || stage2_samples.empty()) {
            throw std::runtime_error("No usable car samples found for tuning");
        }

        cout << "Loaded samples: stage1=" << stage1_samples.size()
             << " stage2=" << stage2_samples.size() << endl;

        auto candidates = generate_candidates(args);
        auto stage1 = run_stage("Stage 1", stage1_samples, candidates, args);
        auto stage2_candidates = keep_top(stage1, args.keep_stage1);
        auto stage2 = run_stage("Stage 2", stage2_samples, stage2_candidates, args);
        auto final_candidates = keep_top(stage2, args.keep_stage2);

        std::vector<Sample> final_samples;
        if (args.final_full) {
            cout << "Loading full-dataset final samples..." << endl;
            Args full_args = args;
            full_args.stage2_frames = 0;
            final_samples = build_stage2_samples(seq_root, full_args);
        } else {
            final_samples = build_fusion_samples(seq_root, args);
            if (final_samples.empty()) {
                cout << "No fusion samples available; final stage will reuse stage2 samples." << endl;
                final_samples = stage2_samples;
            }
        }
        if (final_samples.empty()) throw std::runtime_error("No usable final samples found for tuning");
        cout << "Loaded final samples=" << final_samples.size() << endl;

        auto final_stage = run_stage("Final", final_samples, final_candidates, args);

        const auto report_dir = args.report_dir.empty() ? default_report_dir(argv[0]) : std::filesystem::path(args.report_dir);
        write_reports(args, report_dir, stage1, stage2, final_stage);

        if (!final_stage.empty()) {
            const auto& best = final_stage.front();
            cout << "Recommended: tol=" << std::fixed << std::setprecision(4) << best.candidate.params.tolerance
                 << " min_cluster_size=" << best.candidate.params.min_cluster_size
                 << " max_n=" << best.candidate.params.max_n
                 << " PQ=" << std::setprecision(2) << pq(best.metrics) * 100.0
                 << " time=" << best.metrics.time_ms << " ms" << endl;
        }
    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }
    return 0;
}
