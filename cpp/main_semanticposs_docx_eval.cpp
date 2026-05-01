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
#include "EC.h"
#include "RG.h"
#include "FECunion.h"
#include "FEC_Union.h"

using std::cerr;
using std::cout;
using std::endl;

namespace {

constexpr std::uint16_t kCar = 7;
constexpr std::uint16_t kTrunk = 8;
constexpr std::uint16_t kPlants = 9;
constexpr std::uint16_t kBuilding = 15;
constexpr std::uint16_t kGround = 22;

struct Args {
    std::string root = "/mnt/d/semanticposs";
    std::string report_dir;
    std::vector<std::string> algorithms = {"FECunion", "EC", "RG", "FEC"};
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int min_gt_points = 30;
    int speed_repeat = 7;
    int fusion_window = 10;
    int fusion_samples = 32;
    int fusion_stride = 15;
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

struct SpeedResult {
    double mean = 0.0;
    double stdev = 0.0;
    double min = 0.0;
    double max = 0.0;
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

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << endl;
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };
        if (key == "--root") {
            auto v = need(key); if (!v) return false; args.root = *v;
        } else if (key == "--report-dir") {
            auto v = need(key); if (!v) return false; args.report_dir = *v;
        } else if (key == "--alg") {
            auto v = need(key); if (!v) return false; args.algorithms = split_csv(*v);
        } else if (key == "--tol") {
            auto v = need(key); if (!v) return false; args.tolerance = std::stod(*v);
        } else if (key == "--min-cluster-size") {
            auto v = need(key); if (!v) return false; args.min_cluster_size = std::stoi(*v);
        } else if (key == "--max-n") {
            auto v = need(key); if (!v) return false; args.max_n = std::stoi(*v);
        } else if (key == "--min-gt-points") {
            auto v = need(key); if (!v) return false; args.min_gt_points = std::stoi(*v);
        } else if (key == "--speed-repeat") {
            auto v = need(key); if (!v) return false; args.speed_repeat = std::stoi(*v);
        } else {
            cerr << "Unknown argument: " << key << endl;
            return false;
        }
    }
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

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_bin(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open point cloud: " + path.string());
    struct XYZI { float x, y, z, intensity; };
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<XYZI> raw(static_cast<std::size_t>(size / sizeof(XYZI)));
    in.read(reinterpret_cast<char*>(raw.data()), size);
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(raw.size());
    for (const auto& p : raw) cloud->push_back({p.x, p.y, p.z});
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
        const int frame_index = std::stoi(frame_id);
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
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
    int min_cluster_size,
    double tolerance,
    int max_n
) {
    if (algorithm == "FECunion") return FECunion(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "EC") return EC(cloud, tolerance, min_cluster_size);
    if (algorithm == "RG") return RG(cloud, min_cluster_size, 30, 3.0f, 1.0f);
    if (algorithm == "FEC") return FEC(cloud, min_cluster_size, tolerance, max_n);
    if (algorithm == "FEC_Union") return FEC_Union(cloud, min_cluster_size, tolerance, max_n);
    throw std::runtime_error("Unknown algorithm: " + algorithm);
}

static EvalResult evaluate_sample(
    const Sample& sample,
    const std::string& algorithm,
    const std::set<std::uint16_t>& gt_semantics,
    const Args& args
) {
    std::unordered_map<std::uint32_t, int> gt_key_to_idx;
    std::vector<std::vector<int>> gt_lists;
    std::vector<int> point_to_gt(sample.cloud->size(), -1);
    for (int i = 0; i < static_cast<int>(sample.labels.size()); ++i) {
        const std::uint16_t semantic = static_cast<std::uint16_t>(sample.labels[i] & 0xffffu);
        const std::uint16_t instance = static_cast<std::uint16_t>(sample.labels[i] >> 16);
        if (semantic == 0 || instance == 0 || !contains_semantic(gt_semantics, semantic)) continue;
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
    auto clusters = run_algorithm(cloud, algorithm, args.min_cluster_size, args.tolerance, args.max_n);
    auto t1 = std::chrono::steady_clock::now();
    std::cout.rdbuf(old);

    std::vector<double> best_iou(filtered_gt.size(), 0.0);
    std::vector<std::tuple<double, int, int>> candidates;
    for (int pred_idx = 0; pred_idx < static_cast<int>(clusters.size()); ++pred_idx) {
        std::unordered_map<int, int> overlap;
        for (int point_idx : clusters[pred_idx].indices) {
            if (point_idx < 0 || point_idx >= static_cast<int>(point_to_gt.size())) continue;
            int gt = point_to_gt[point_idx];
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
static double fps(double ms) { return ms > 1e-12 ? 1000.0 / ms : 0.0; }

static SpeedResult speed_benchmark(const Sample& sample, const std::string& algorithm, const Args& args) {
    std::vector<double> times;
    for (int i = 0; i < args.speed_repeat; ++i) {
        auto r = evaluate_sample(sample, algorithm, {}, args);
        times.push_back(r.time_ms);
    }
    SpeedResult s;
    s.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    s.min = *std::min_element(times.begin(), times.end());
    s.max = *std::max_element(times.begin(), times.end());
    double sqsum = 0.0;
    for (double t : times) sqsum += (t - s.mean) * (t - s.mean);
    s.stdev = std::sqrt(sqsum / times.size());
    return s;
}

static std::filesystem::path default_report_dir(const char* argv0) {
    std::filesystem::path exe = std::filesystem::weakly_canonical(std::filesystem::absolute(argv0));
    return exe.parent_path().parent_path().parent_path() / "reports";
}

static void write_metric_row(std::ostream& out, const std::string& alg, const EvalResult& r) {
    out << "|" << alg
        << "|" << std::fixed << std::setprecision(2) << r.time_ms
        << "|" << pq(r) * 100.0
        << "|" << sq(r) * 100.0
        << "|" << rq(r) * 100.0
        << "|" << rc50(r) * 100.0
        << "|" << miou(r) * 100.0
        << "|" << r.samples
        << "|" << r.points
        << "|" << r.gt
        << "|" << r.pred
        << "|\n";
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    try {
        const auto seq_root = sequences_root(args.root);
        const auto seqs = list_sequences(seq_root);

        std::unordered_map<std::string, SpeedResult> speed;
        std::unordered_map<std::string, EvalResult> multiseq_car;
        std::unordered_map<std::string, EvalResult> fusion_car;
        std::unordered_map<std::string, EvalResult> hard_scene;

        auto seq00 = seq_root / "00";
        auto speed_sample = make_sample(seq00, {"000000"}, {}, true, false, "seq00_000000_noground");
        cout << "Section 1: single-frame speed" << endl;
        for (const auto& alg : args.algorithms) {
            speed[alg] = speed_benchmark(speed_sample, alg, args);
            cout << alg << " mean=" << speed[alg].mean << " ms" << endl;
        }

        cout << "Section 2: multisequence car accuracy" << endl;
        for (const auto& alg : args.algorithms) {
            std::vector<EvalResult> per_frame;
            for (const auto& seq : seqs) {
                const auto sroot = seq_root / seq;
                for (const auto& frame : list_frames(sroot)) {
                    auto sample = make_sample(sroot, {frame}, {kCar}, false, false, seq + "_" + frame + "_car");
                    if (sample.used_points == 0) continue;
                    per_frame.push_back(evaluate_sample(sample, alg, {kCar}, args));
                }
            }
            multiseq_car[alg] = merge_results(per_frame);
            cout << alg << " PQ=" << pq(multiseq_car[alg]) * 100.0 << endl;
        }

        cout << "Section 3: fusion car accuracy" << endl;
        auto frames00 = list_frames(seq00);
        for (const auto& alg : args.algorithms) {
            std::vector<EvalResult> samples;
            for (int i = 0; i < args.fusion_samples; ++i) {
                const int start = i * args.fusion_stride;
                if (start + args.fusion_window > static_cast<int>(frames00.size())) break;
                std::vector<std::string> window(frames00.begin() + start, frames00.begin() + start + args.fusion_window);
                auto sample = make_sample(seq00, window, {kCar}, false, true, "fusion_car");
                if (sample.used_points == 0) continue;
                samples.push_back(evaluate_sample(sample, alg, {kCar}, args));
            }
            fusion_car[alg] = merge_results(samples);
            cout << alg << " PQ=" << pq(fusion_car[alg]) * 100.0 << endl;
        }

        cout << "Section 4: hard car + tree/building scene" << endl;
        const std::vector<int> targets = {1000000, 2000000, 3000000};
        for (const auto& alg : args.algorithms) {
            std::vector<EvalResult> samples;
            int start = 0;
            for (int target : targets) {
                std::vector<std::string> chosen;
                int points = 0;
                for (int idx = start; idx < static_cast<int>(frames00.size()) && points < target; ++idx) {
                    chosen.push_back(frames00[idx]);
                    auto probe = make_sample(seq00, {frames00[idx]}, {kCar, kTrunk, kPlants, kBuilding}, false, false, "probe");
                    points += probe.used_points;
                }
                start += 30;
                auto sample = make_sample(seq00, chosen, {kCar, kTrunk, kPlants, kBuilding}, false, true, "hard");
                if (sample.used_points == 0) continue;
                samples.push_back(evaluate_sample(sample, alg, {kCar}, args));
            }
            hard_scene[alg] = merge_results(samples);
            cout << alg << " PQ=" << pq(hard_scene[alg]) * 100.0 << endl;
        }

        const auto report_dir = args.report_dir.empty() ? default_report_dir(argv[0]) : std::filesystem::path(args.report_dir);
        std::filesystem::create_directories(report_dir);
        const auto report_path = report_dir / "semanticposs_full_experiment_report.md";
        std::ofstream out(report_path);
        if (!out) throw std::runtime_error("Failed to write report: " + report_path.string());

        out << "# SemanticPOSS Full Experiment Report\n\n";
        out << "## Setup\n\n";
        out << "- Dataset root: `" << args.root << "`\n";
        out << "- Sequences: `00, 01, 02, 03, 04, 05`\n";
        out << "- Algorithms: `FECunion`, `EC`, `RG`, `FEC`\n";
        out << "- Parameters: `tol=" << args.tolerance << "`, `min_cluster_size=" << args.min_cluster_size
            << "`, `max_n=" << args.max_n << "`, `min_gt_points=" << args.min_gt_points << "`\n";
        out << "- Semantic ids: `car=7`, `trunk=8`, `plants=9`, `building=15`, `ground=22`\n\n";

        out << "## 1. Single-Frame Fixed-Parameter Speed\n\n";
        out << "- Input: sequence `00`, frame `000000`, ground removed, whole frame sent to clustering.\n";
        out << "- Points: `" << speed_sample.used_points << "`\n\n";
        out << "| Algorithm | Mean(ms) | Std(ms) | Min(ms) | Max(ms) | FPS |\n";
        out << "|---|---:|---:|---:|---:|---:|\n";
        for (const auto& alg : args.algorithms) {
            const auto& s = speed[alg];
            out << "|" << alg << "|" << std::fixed << std::setprecision(2) << s.mean << "|" << s.stdev
                << "|" << s.min << "|" << s.max << "|" << fps(s.mean) << "|\n";
        }

        out << "\n## 2. Multi-Sequence Single-Frame Car Accuracy\n\n";
        out << "- Input: all frames from six SemanticPOSS sequences; only `car` points are sent to clustering.\n";
        out << "- Note: the original report had 11 sequences; this dataset copy contains 6 sequences.\n\n";
        out << "| Algorithm | Time(ms) | PQ | SQ | RQ | RC50 | mIoU | Samples | Points | GT | Pred |\n";
        out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& alg : args.algorithms) write_metric_row(out, alg, multiseq_car[alg]);

        out << "\n## 3. Multi-Frame Fusion Fixed-Parameter Car Accuracy\n\n";
        out << "- Input: sequence `00`, 32 fused samples; each sample fuses 10 consecutive frames with poses.\n";
        out << "- Only `car` points are sent to clustering.\n\n";
        out << "| Algorithm | Time(ms) | PQ | SQ | RQ | RC50 | mIoU | Samples | Points | GT | Pred |\n";
        out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& alg : args.algorithms) write_metric_row(out, alg, fusion_car[alg]);

        out << "\n## 4. Hard Scene Supplement: Car + Tree/Building Interference\n\n";
        out << "- Input: sequence `00`, three fused samples targeting about 1M, 2M, and 3M selected points.\n";
        out << "- Clustering input: `car + trunk + plants + building`; scoring target: `car` instances.\n\n";
        out << "| Algorithm | Time(ms) | PQ | SQ | RQ | RC50 | mIoU | Samples | Points | GT | Pred |\n";
        out << "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& alg : args.algorithms) write_metric_row(out, alg, hard_scene[alg]);

        out << "\n## Metric Notes\n\n";
        out << "- `PQ = SQ * RQ`.\n";
        out << "- `SQ` is the mean IoU over one-to-one matches with `IoU >= 0.5`.\n";
        out << "- `RQ = TP / (TP + 0.5 FP + 0.5 FN)`.\n";
        out << "- `RC50 = TP / (TP + FN)` at `IoU >= 0.5`.\n";
        out << "- `mIoU` is the mean best-overlap IoU over GT instances.\n";

        cout << "Report written: " << report_path.string() << endl;
    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }
    return 0;
}
