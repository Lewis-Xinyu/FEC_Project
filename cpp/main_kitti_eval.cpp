#pragma warning(disable:4996)
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
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

using std::cout;
using std::cerr;
using std::endl;

struct Args {
    std::string root = "/mnt/f/datasets/kitti";
    std::string frame = "000000";
    std::string algorithm = "FEC_Union_Block_new2";
    double tolerance = 0.2;
    int min_cluster_size = 100;
    int max_n = 50;
    int min_gt_points = 30;
    bool use_reduced = false;
    bool evaluate_all = false;
    int limit = -1;
};

struct KittiCalib {
    Eigen::Matrix3d R0_rect = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 4> Tr_velo_to_cam = Eigen::Matrix<double, 3, 4>::Zero();
    bool valid = false;
};

struct KittiLabel {
    std::string type;
    double h = 0.0;
    double w = 0.0;
    double l = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double ry = 0.0;
};

struct FrameEval {
    std::string frame_id;
    int total_points = 0;
    int gt_objects = 0;
    int predicted_clusters = 0;
    int matched_03 = 0;
    int matched_05 = 0;
    double mean_best_iou = 0.0;
    double mean_best_recall = 0.0;
    double mean_best_precision = 0.0;
    double clustering_ms = 0.0;
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
    cout << "KITTI clustering evaluation\n"
         << "Usage:\n"
         << "  ./cpp/build/kitti_eval_run [options]\n\n"
         << "Options:\n"
         << "  --root PATH              KITTI root containing calib/ label_2/ velodyne/\n"
         << "  --frame ID               Frame id, e.g. 000123\n"
         << "  --alg NAME               Algorithm name, same as fec_run\n"
         << "  --tol VALUE              Distance tolerance\n"
         << "  --min-cluster-size N     Minimum predicted cluster size\n"
         << "  --max-n N                Max neighbors for radius search\n"
         << "  --min-gt-points N        Ignore GT objects with fewer LiDAR points than this\n"
         << "  --use-reduced 0|1        Use velodyne_reduced/ instead of velodyne/\n"
         << "  --all 0|1                Evaluate all frames under the selected folder\n"
         << "  --limit N                Max number of frames when --all 1\n"
         << "  --help                   Show this help\n\n"
         << "Examples:\n"
         << "  ./cpp/build/kitti_eval_run --root /mnt/f/datasets/kitti --frame 000046 --alg FEC_Union_Block_new2 --tol 0.2\n"
         << "  ./cpp/build/kitti_eval_run --root /mnt/f/datasets/kitti --all 1 --limit 20 --alg FEC_Union_Block_new2 --tol 0.2\n";
}

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            print_help();
            return false;
        }
        auto need_value = [&](const std::string& name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                cerr << "Missing value for " << name << endl;
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };

        if (key == "--root") {
            auto value = need_value(key);
            if (!value) return false;
            args.root = *value;
        } else if (key == "--frame") {
            auto value = need_value(key);
            if (!value) return false;
            args.frame = *value;
        } else if (key == "--alg") {
            auto value = need_value(key);
            if (!value) return false;
            args.algorithm = *value;
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
        } else if (key == "--use-reduced") {
            auto value = need_value(key);
            if (!value) return false;
            args.use_reduced = parse_bool(*value, args.use_reduced);
        } else if (key == "--all") {
            auto value = need_value(key);
            if (!value) return false;
            args.evaluate_all = parse_bool(*value, args.evaluate_all);
        } else if (key == "--limit") {
            auto value = need_value(key);
            if (!value) return false;
            args.limit = std::stoi(*value);
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
    const std::string& current_algorithm,
    int min_cluster_size,
    double tolerance,
    int max_n
) {
    if (current_algorithm == "FEC") return FEC(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Block") return FEC_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC1") return FEC1(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC1_1") return FEC1_1(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FECunion") return FECunion(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "EC") return EC(cloud, tolerance, min_cluster_size);
    if (current_algorithm == "EC_Block") return EC_Block(cloud, tolerance, min_cluster_size);
    if (current_algorithm == "RG") return RG(cloud, 0.785, 0.5, 30, min_cluster_size);
    if (current_algorithm == "FEC_Union") return FEC_Union(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Block") return FEC_Union_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Block_new") return FEC_Union_Block_new(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Block_new2") return FEC_Union_Block_new2(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FECunion_Block") return FECunion_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC_Union_Grid_Block") return FEC_Union_Grid_Block(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "Voxel_FEC1") return Voxel_FEC1(cloud, min_cluster_size, tolerance, max_n);
    if (current_algorithm == "FEC1_Block") return FEC1_Block(cloud, min_cluster_size, tolerance, max_n);

    throw std::runtime_error("Unknown algorithm: " + current_algorithm);
}

static pcl::PointCloud<pcl::PointXYZ>::Ptr load_kitti_bin_cloud(const std::filesystem::path& bin_path) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open point cloud: " + bin_path.string());
    }

    struct PointXYZI {
        float x, y, z, intensity;
    };

    in.seekg(0, std::ios::end);
    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size % static_cast<std::streamsize>(sizeof(PointXYZI)) != 0) {
        throw std::runtime_error("Unexpected KITTI .bin size: " + bin_path.string());
    }

    const std::size_t point_count = static_cast<std::size_t>(file_size / sizeof(PointXYZI));
    std::vector<PointXYZI> raw(point_count);
    in.read(reinterpret_cast<char*>(raw.data()), file_size);

    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(point_count);
    for (const auto& p : raw) {
        cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    cloud->width = static_cast<std::uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

static std::vector<double> parse_calib_values(const std::string& line) {
    std::size_t pos = line.find(':');
    if (pos == std::string::npos) return {};
    std::stringstream ss(line.substr(pos + 1));
    std::vector<double> values;
    double v = 0.0;
    while (ss >> v) {
        values.push_back(v);
    }
    return values;
}

static KittiCalib load_kitti_calib(const std::filesystem::path& calib_path) {
    std::ifstream in(calib_path);
    if (!in) {
        throw std::runtime_error("Failed to open calib file: " + calib_path.string());
    }

    KittiCalib calib;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("R0_rect:", 0) == 0 || line.rfind("R_rect:", 0) == 0) {
            auto values = parse_calib_values(line);
            if (values.size() == 9) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        calib.R0_rect(r, c) = values[r * 3 + c];
                    }
                }
            }
        } else if (line.rfind("Tr_velo_to_cam:", 0) == 0) {
            auto values = parse_calib_values(line);
            if (values.size() == 12) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        calib.Tr_velo_to_cam(r, c) = values[r * 4 + c];
                    }
                }
                calib.valid = true;
            }
        }
    }

    if (!calib.valid) {
        throw std::runtime_error("Missing Tr_velo_to_cam in calib file: " + calib_path.string());
    }
    return calib;
}

static std::vector<KittiLabel> load_kitti_labels(const std::filesystem::path& label_path) {
    std::ifstream in(label_path);
    if (!in) {
        throw std::runtime_error("Failed to open label file: " + label_path.string());
    }

    std::vector<KittiLabel> labels;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::stringstream ss(line);
        KittiLabel obj;
        double truncated = 0.0;
        int occluded = 0;
        double alpha = 0.0;
        double bbox_left = 0.0, bbox_top = 0.0, bbox_right = 0.0, bbox_bottom = 0.0;
        ss >> obj.type >> truncated >> occluded >> alpha
           >> bbox_left >> bbox_top >> bbox_right >> bbox_bottom
           >> obj.h >> obj.w >> obj.l >> obj.x >> obj.y >> obj.z >> obj.ry;

        if (!ss.fail()) {
            labels.push_back(obj);
        }
    }
    return labels;
}

static std::vector<Eigen::Vector3d> transform_points_to_rect(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const KittiCalib& calib
) {
    std::vector<Eigen::Vector3d> rect_points;
    rect_points.reserve(cloud->size());

    for (const auto& p : cloud->points) {
        Eigen::Vector4d velo(p.x, p.y, p.z, 1.0);
        Eigen::Vector3d cam = calib.Tr_velo_to_cam * velo;
        Eigen::Vector3d rect = calib.R0_rect * cam;
        rect_points.push_back(rect);
    }
    return rect_points;
}

static bool should_use_label_type(const std::string& type) {
    return !(type == "DontCare" || type == "Misc");
}

static bool point_in_kitti_box(const Eigen::Vector3d& point_rect, const KittiLabel& box) {
    Eigen::Vector3d center(box.x, box.y, box.z);
    Eigen::Vector3d q = point_rect - center;

    const double c = std::cos(box.ry);
    const double s = std::sin(box.ry);
    const double local_x = c * q.x() - s * q.z();
    const double local_y = q.y();
    const double local_z = s * q.x() + c * q.z();

    return std::abs(local_x) <= box.w * 0.5 &&
           local_y >= -box.h && local_y <= 0.0 &&
           std::abs(local_z) <= box.l * 0.5;
}

static std::vector<std::string> list_frame_ids(const std::filesystem::path& root, bool use_reduced) {
    std::filesystem::path velodyne_dir = root / (use_reduced ? "velodyne_reduced" : "velodyne");
    if (!std::filesystem::exists(velodyne_dir)) {
        throw std::runtime_error("Velodyne directory not found: " + velodyne_dir.string());
    }

    std::vector<std::string> ids;
    for (const auto& entry : std::filesystem::directory_iterator(velodyne_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".bin") continue;
        ids.push_back(entry.path().stem().string());
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

static FrameEval evaluate_frame(const Args& args, const std::string& frame_id) {
    const std::filesystem::path root(args.root);
    const std::filesystem::path bin_path = root / (args.use_reduced ? "velodyne_reduced" : "velodyne") / (frame_id + ".bin");
    const std::filesystem::path label_path = root / "label_2" / (frame_id + ".txt");
    const std::filesystem::path calib_path = root / "calib" / (frame_id + ".txt");

    auto cloud = load_kitti_bin_cloud(bin_path);
    KittiCalib calib = load_kitti_calib(calib_path);
    auto labels = load_kitti_labels(label_path);
    auto rect_points = transform_points_to_rect(cloud, calib);

    std::vector<KittiLabel> filtered_labels;
    filtered_labels.reserve(labels.size());
    for (const auto& label : labels) {
        if (should_use_label_type(label.type)) {
            filtered_labels.push_back(label);
        }
    }

    std::vector<std::vector<int>> gt_point_lists;
    std::vector<KittiLabel> valid_gt_labels;
    gt_point_lists.reserve(filtered_labels.size());
    valid_gt_labels.reserve(filtered_labels.size());

    for (const auto& label : filtered_labels) {
        std::vector<int> indices;
        for (int i = 0; i < static_cast<int>(rect_points.size()); ++i) {
            if (point_in_kitti_box(rect_points[i], label)) {
                indices.push_back(i);
            }
        }
        if (static_cast<int>(indices.size()) >= args.min_gt_points) {
            gt_point_lists.push_back(std::move(indices));
            valid_gt_labels.push_back(label);
        }
    }

    auto cluster_begin = std::chrono::steady_clock::now();
    auto clusters = run_algorithm(cloud, args.algorithm, args.min_cluster_size, args.tolerance, args.max_n);
    auto cluster_end = std::chrono::steady_clock::now();
    const double clustering_ms = std::chrono::duration<double, std::milli>(cluster_end - cluster_begin).count();

    std::vector<int> point_to_gt(cloud->size(), -1);
    for (int gt_idx = 0; gt_idx < static_cast<int>(gt_point_lists.size()); ++gt_idx) {
        for (int point_idx : gt_point_lists[gt_idx]) {
            if (point_to_gt[point_idx] == -1) {
                point_to_gt[point_idx] = gt_idx;
            }
        }
    }

    std::vector<double> best_iou(gt_point_lists.size(), 0.0);
    std::vector<double> best_recall(gt_point_lists.size(), 0.0);
    std::vector<double> best_precision(gt_point_lists.size(), 0.0);

    for (const auto& cluster : clusters) {
        if (cluster.indices.empty()) continue;

        std::unordered_map<int, int> overlap_count;
        for (int point_idx : cluster.indices) {
            if (point_idx < 0 || point_idx >= static_cast<int>(point_to_gt.size())) continue;
            int gt_idx = point_to_gt[point_idx];
            if (gt_idx >= 0) {
                ++overlap_count[gt_idx];
            }
        }

        for (const auto& [gt_idx, inter] : overlap_count) {
            const int gt_size = static_cast<int>(gt_point_lists[gt_idx].size());
            const int pred_size = static_cast<int>(cluster.indices.size());
            const double union_size = static_cast<double>(gt_size + pred_size - inter);
            const double iou = union_size > 0.0 ? static_cast<double>(inter) / union_size : 0.0;
            if (iou > best_iou[gt_idx]) {
                best_iou[gt_idx] = iou;
                best_recall[gt_idx] = gt_size > 0 ? static_cast<double>(inter) / gt_size : 0.0;
                best_precision[gt_idx] = pred_size > 0 ? static_cast<double>(inter) / pred_size : 0.0;
            }
        }
    }

    FrameEval result;
    result.frame_id = frame_id;
    result.total_points = static_cast<int>(cloud->size());
    result.gt_objects = static_cast<int>(gt_point_lists.size());
    result.predicted_clusters = static_cast<int>(clusters.size());
    result.clustering_ms = clustering_ms;

    if (!gt_point_lists.empty()) {
        for (int i = 0; i < static_cast<int>(gt_point_lists.size()); ++i) {
            result.mean_best_iou += best_iou[i];
            result.mean_best_recall += best_recall[i];
            result.mean_best_precision += best_precision[i];
            if (best_iou[i] >= 0.3) ++result.matched_03;
            if (best_iou[i] >= 0.5) ++result.matched_05;
        }
        const double denom = static_cast<double>(gt_point_lists.size());
        result.mean_best_iou /= denom;
        result.mean_best_recall /= denom;
        result.mean_best_precision /= denom;
    }

    cout << std::fixed << std::setprecision(3)
         << "Eval frame=" << frame_id
         << " points=" << result.total_points
         << " gt=" << result.gt_objects
         << " pred_clusters=" << result.predicted_clusters
         << " mIoU=" << result.mean_best_iou
         << " mRecall=" << result.mean_best_recall
         << " mPrecision=" << result.mean_best_precision
         << " matched@0.3=" << result.matched_03
         << " matched@0.5=" << result.matched_05
         << " cluster_wall=" << result.clustering_ms << " ms"
         << "\n";

    return result;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 0;
    }

    try {
        const std::filesystem::path root(args.root);
        if (!std::filesystem::exists(root / "calib") ||
            !std::filesystem::exists(root / "label_2") ||
            !std::filesystem::exists(root / (args.use_reduced ? "velodyne_reduced" : "velodyne"))) {
            cerr << "KITTI root is incomplete: " << root << endl;
            cerr << "Expected directories: calib/, label_2/, "
                 << (args.use_reduced ? "velodyne_reduced/" : "velodyne/") << endl;
            cerr << "If you only have training.zip, extract it first into a folder like /mnt/f/datasets/kitti" << endl;
            return 1;
        }

        cout << "KITTI Eval\n";
        cout << "Root: " << root.string() << "\n";
        cout << "Algorithm: " << args.algorithm << "\n";
        cout << "Tolerance: " << args.tolerance << "\n";
        cout << "Min cluster size: " << args.min_cluster_size << "\n";
        cout << "Min GT points: " << args.min_gt_points << "\n";
        cout << "Point source: " << (args.use_reduced ? "velodyne_reduced" : "velodyne") << "\n";

        std::vector<std::string> frame_ids;
        if (args.evaluate_all) {
            frame_ids = list_frame_ids(root, args.use_reduced);
            if (args.limit > 0 && args.limit < static_cast<int>(frame_ids.size())) {
                frame_ids.resize(args.limit);
            }
        } else {
            frame_ids = {args.frame};
        }

        cout << "Frames to evaluate: " << frame_ids.size() << "\n\n";

        std::vector<FrameEval> results;
        results.reserve(frame_ids.size());
        for (const auto& frame_id : frame_ids) {
            results.push_back(evaluate_frame(args, frame_id));
        }

        if (results.empty()) {
            cout << "No frames evaluated.\n";
            return 0;
        }

        double avg_miou = 0.0;
        double avg_recall = 0.0;
        double avg_precision = 0.0;
        double avg_cluster_ms = 0.0;
        int total_gt = 0;
        int total_pred = 0;
        int total_match_03 = 0;
        int total_match_05 = 0;

        for (const auto& r : results) {
            avg_miou += r.mean_best_iou;
            avg_recall += r.mean_best_recall;
            avg_precision += r.mean_best_precision;
            avg_cluster_ms += r.clustering_ms;
            total_gt += r.gt_objects;
            total_pred += r.predicted_clusters;
            total_match_03 += r.matched_03;
            total_match_05 += r.matched_05;
        }

        const double denom = static_cast<double>(results.size());
        avg_miou /= denom;
        avg_recall /= denom;
        avg_precision /= denom;
        avg_cluster_ms /= denom;

        cout << "\nSummary\n";
        cout << std::fixed << std::setprecision(3)
             << "frames=" << results.size()
             << " total_gt=" << total_gt
             << " total_pred_clusters=" << total_pred
             << " avg_mIoU=" << avg_miou
             << " avg_mRecall=" << avg_recall
             << " avg_mPrecision=" << avg_precision
             << " gt_match_rate@0.3=" << (total_gt > 0 ? static_cast<double>(total_match_03) / total_gt : 0.0)
             << " gt_match_rate@0.5=" << (total_gt > 0 ? static_cast<double>(total_match_05) / total_gt : 0.0)
             << " avg_cluster_wall=" << avg_cluster_ms << " ms"
             << "\n";
    } catch (const std::exception& ex) {
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}
