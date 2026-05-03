#pragma once

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace semantic_kitti_paths {

inline bool looks_like_sequences_root(const std::filesystem::path& path) {
    return std::filesystem::exists(path / "00" / "velodyne");
}

inline std::filesystem::path try_resolve_sequences_root(const std::filesystem::path& root) {
    if (root.empty()) return {};
    if (looks_like_sequences_root(root)) return root;
    if (looks_like_sequences_root(root / "sequences")) return root / "sequences";
    if (looks_like_sequences_root(root / "dataset" / "sequences")) return root / "dataset" / "sequences";
    return {};
}

inline void append_candidate_if_present(
    std::vector<std::filesystem::path>& candidates,
    const std::filesystem::path& path
) {
    if (!path.empty()) candidates.push_back(path);
}

inline std::vector<std::filesystem::path> candidate_roots() {
    std::vector<std::filesystem::path> candidates;

    if (const char* env = std::getenv("FEC_SEMANTIC_KITTI_ROOT")) {
        append_candidate_if_present(candidates, std::filesystem::path(env));
    }
    if (const char* env = std::getenv("SEMANTIC_KITTI_ROOT")) {
        append_candidate_if_present(candidates, std::filesystem::path(env));
    }

    append_candidate_if_present(candidates, "/Volumes/曹鑫宇的ssd/datasets/semantic_kitti");
    append_candidate_if_present(candidates, "/Volumes/曹鑫宇的ssd/datasets/semantic_kitti/dataset");
    append_candidate_if_present(candidates, "/mnt/f/datasets/semantic_kitti");
    append_candidate_if_present(candidates, "/mnt/f/datasets/semantic_kitti/dataset");
    append_candidate_if_present(candidates, "/mnt/f/datasets/kitti");
    append_candidate_if_present(candidates, "/mnt/f/datasets/kitti/dataset");

    const std::filesystem::path volumes("/Volumes");
    if (std::filesystem::exists(volumes)) {
        for (const auto& entry : std::filesystem::directory_iterator(volumes)) {
            if (!entry.is_directory()) continue;
            append_candidate_if_present(candidates, entry.path() / "datasets" / "semantic_kitti");
            append_candidate_if_present(candidates, entry.path() / "datasets" / "semantic_kitti" / "dataset");
            append_candidate_if_present(candidates, entry.path() / "semantic_kitti");
            append_candidate_if_present(candidates, entry.path() / "semantic_kitti" / "dataset");
        }
    }

    return candidates;
}

inline std::filesystem::path resolve_sequences_root(const std::optional<std::filesystem::path>& preferred_root) {
    if (preferred_root && !preferred_root->empty()) {
        const auto resolved = try_resolve_sequences_root(*preferred_root);
        if (!resolved.empty()) return resolved;
        throw std::runtime_error(
            "SemanticKITTI root is incomplete: " + preferred_root->string() +
            " (expected sequences/ or dataset/sequences/)"
        );
    }

    std::vector<std::string> tried;
    for (const auto& candidate : candidate_roots()) {
        tried.push_back(candidate.string());
        const auto resolved = try_resolve_sequences_root(candidate);
        if (!resolved.empty()) return resolved;
    }

    std::ostringstream oss;
    oss << "Could not auto-detect SemanticKITTI root. Tried:";
    for (const auto& item : tried) oss << "\n  - " << item;
    oss << "\nSet --root PATH or export FEC_SEMANTIC_KITTI_ROOT.";
    throw std::runtime_error(oss.str());
}

}  // namespace semantic_kitti_paths
