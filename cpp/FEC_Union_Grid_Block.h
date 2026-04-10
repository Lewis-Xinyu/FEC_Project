#pragma once
#ifndef PCL_SEGEMENT_FEC_UNION_GRID_BLOCK_H
#define PCL_SEGEMENT_FEC_UNION_GRID_BLOCK_H

#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <numeric>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

struct FECUGB_PointIndexTag {
    int point_index;
    int tag;
};

inline bool FECUGB_TagLess(const FECUGB_PointIndexTag& a, const FECUGB_PointIndexTag& b) {
    return a.tag < b.tag;
}

struct FECUGB_BlockKey {
    int x, y, z;
    bool operator==(const FECUGB_BlockKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct FECUGB_BlockKeyHash {
    std::size_t operator()(const FECUGB_BlockKey& k) const noexcept {
        std::size_t h1 = std::hash<int>{}(k.x);
        std::size_t h2 = std::hash<int>{}(k.y);
        std::size_t h3 = std::hash<int>{}(k.z);
        return h1 ^ (h2 << 1) ^ (h3 << 7);
    }
};

struct FECUGB_LocalCluster {
    std::vector<int> all_global_indices;
    std::vector<int> owner_global_indices;
};

struct FECUGB_LocalStats {
    double build_ms = 0.0;
    double search_ms = 0.0;
    double merge_ms = 0.0;
    double final_ms = 0.0;
};

struct FECUGB_BlockData {
    FECUGB_BlockKey key{};
    std::vector<int> owner_indices;
    std::vector<int> expanded_indices;
};

class FECUGB_DisjointSet {
public:
    FECUGB_DisjointSet() = default;
    explicit FECUGB_DisjointSet(int n) { reset(n); }

    void reset(int n) {
        parent.resize(n);
        rank.assign(n, 0);
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;
        if (rank[ra] < rank[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) ++rank[ra];
    }

private:
    std::vector<int> parent;
    std::vector<int> rank;
};

inline std::vector<FECUGB_LocalCluster> FECUGB_LocalClusterOnly(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const std::vector<int>& expanded_global_indices,
    const std::vector<int>& owner_global_indices,
    double tolorance,
    int max_n,
    FECUGB_LocalStats& stats
) {
    using Clock = std::chrono::steady_clock;

    std::vector<FECUGB_LocalCluster> local_clusters;
    if (expanded_global_indices.empty()) return local_clusters;

    auto tb0 = Clock::now();

    const int n = static_cast<int>(expanded_global_indices.size());
    const float cell_size = static_cast<float>(std::max(tolorance, 1e-6));
    std::unordered_map<FECUGB_BlockKey, std::vector<int>, FECUGB_BlockKeyHash> grid;
    grid.reserve(n / 4 + 1);

    auto point_to_cell = [&](const pcl::PointXYZ& pt) -> FECUGB_BlockKey {
        return FECUGB_BlockKey{
            static_cast<int>(std::floor(pt.x / cell_size)),
            static_cast<int>(std::floor(pt.y / cell_size)),
            static_cast<int>(std::floor(pt.z / cell_size))
        };
    };

    for (int local_idx = 0; local_idx < n; ++local_idx) {
        const auto& pt = cloud->points[expanded_global_indices[local_idx]];
        grid[point_to_cell(pt)].push_back(local_idx);
    }

    auto tb1 = Clock::now();
    stats.build_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();

    FECUGB_DisjointSet dsu(n);
    std::vector<bool> searched(n, false);
    std::vector<std::pair<float, int>> neighbor_candidates;
    neighbor_candidates.reserve(std::max(32, max_n * 3));
    const float radius2 = static_cast<float>(tolorance * tolorance);

    for (int i = 0; i < n; ++i) {
        if (searched[i]) continue;

        const auto& query_pt = cloud->points[expanded_global_indices[i]];
        const FECUGB_BlockKey center = point_to_cell(query_pt);

        auto ts0 = Clock::now();
        neighbor_candidates.clear();

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    FECUGB_BlockKey key{center.x + dx, center.y + dy, center.z + dz};
                    auto it = grid.find(key);
                    if (it == grid.end()) continue;

                    for (int local_idx : it->second) {
                        const auto& pt = cloud->points[expanded_global_indices[local_idx]];
                        const float ddx = pt.x - query_pt.x;
                        const float ddy = pt.y - query_pt.y;
                        const float ddz = pt.z - query_pt.z;
                        const float dist2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        if (dist2 <= radius2) {
                            neighbor_candidates.push_back({dist2, local_idx});
                        }
                    }
                }
            }
        }

        std::sort(neighbor_candidates.begin(), neighbor_candidates.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        if (max_n > 0 && static_cast<int>(neighbor_candidates.size()) > max_n) {
            neighbor_candidates.resize(max_n);
        }

        auto ts1 = Clock::now();
        stats.search_ms += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

        auto tm0 = Clock::now();
        for (const auto& candidate : neighbor_candidates) {
            const int idx = candidate.second;
            dsu.unite(i, idx);
            searched[idx] = true;
        }
        auto tm1 = Clock::now();
        stats.merge_ms += std::chrono::duration<double, std::milli>(tm1 - tm0).count();
    }

    auto tf0 = Clock::now();

    std::unordered_set<int> owner_set(owner_global_indices.begin(), owner_global_indices.end());
    std::vector<FECUGB_PointIndexTag> indices_tags(n);
    for (int i = 0; i < n; ++i) {
        indices_tags[i].point_index = i;
        indices_tags[i].tag = dsu.find(i);
    }
    std::sort(indices_tags.begin(), indices_tags.end(), FECUGB_TagLess);

    int begin_index = 0;
    for (int i = 0; i < n; ++i) {
        if (indices_tags[i].tag != indices_tags[begin_index].tag) {
            FECUGB_LocalCluster cluster;
            for (int j = begin_index; j < i; ++j) {
                int local_idx = indices_tags[j].point_index;
                int global_idx = expanded_global_indices[local_idx];
                cluster.all_global_indices.push_back(global_idx);
                if (owner_set.find(global_idx) != owner_set.end()) {
                    cluster.owner_global_indices.push_back(global_idx);
                }
            }
            std::sort(cluster.all_global_indices.begin(), cluster.all_global_indices.end());
            cluster.all_global_indices.erase(
                std::unique(cluster.all_global_indices.begin(), cluster.all_global_indices.end()),
                cluster.all_global_indices.end()
            );
            std::sort(cluster.owner_global_indices.begin(), cluster.owner_global_indices.end());
            cluster.owner_global_indices.erase(
                std::unique(cluster.owner_global_indices.begin(), cluster.owner_global_indices.end()),
                cluster.owner_global_indices.end()
            );
            if (!cluster.all_global_indices.empty()) {
                local_clusters.push_back(std::move(cluster));
            }
            begin_index = i;
        }
    }

    FECUGB_LocalCluster last_cluster;
    for (int j = begin_index; j < n; ++j) {
        int local_idx = indices_tags[j].point_index;
        int global_idx = expanded_global_indices[local_idx];
        last_cluster.all_global_indices.push_back(global_idx);
        if (owner_set.find(global_idx) != owner_set.end()) {
            last_cluster.owner_global_indices.push_back(global_idx);
        }
    }
    std::sort(last_cluster.all_global_indices.begin(), last_cluster.all_global_indices.end());
    last_cluster.all_global_indices.erase(
        std::unique(last_cluster.all_global_indices.begin(), last_cluster.all_global_indices.end()),
        last_cluster.all_global_indices.end()
    );
    std::sort(last_cluster.owner_global_indices.begin(), last_cluster.owner_global_indices.end());
    last_cluster.owner_global_indices.erase(
        std::unique(last_cluster.owner_global_indices.begin(), last_cluster.owner_global_indices.end()),
        last_cluster.owner_global_indices.end()
    );
    if (!last_cluster.all_global_indices.empty()) {
        local_clusters.push_back(std::move(last_cluster));
    }

    auto tf1 = Clock::now();
    stats.final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

    return local_clusters;
}

inline std::vector<pcl::PointIndices> FEC_Union_Grid_Block(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    int min_component_size,
    double tolorance,
    int max_n
) {
    using Clock = std::chrono::steady_clock;

    auto t_total_begin = Clock::now();

    if (!cloud || cloud->empty() || cloud->size() < static_cast<std::size_t>(min_component_size)) {
        return {};
    }

    auto tp0 = Clock::now();

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();

    for (const auto& p : cloud->points) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        min_z = std::min(min_z, p.z);
    }

    const double block_size = std::max(tolorance * 8.0, tolorance + 1e-6);

    auto get_block_key = [&](const pcl::PointXYZ& pt) -> FECUGB_BlockKey {
        return FECUGB_BlockKey{
            static_cast<int>(std::floor((pt.x - min_x) / block_size)),
            static_cast<int>(std::floor((pt.y - min_y) / block_size)),
            static_cast<int>(std::floor((pt.z - min_z) / block_size))
        };
    };

    std::unordered_map<FECUGB_BlockKey, int, FECUGB_BlockKeyHash> block_to_id;
    std::vector<FECUGB_BlockData> blocks;
    blocks.reserve(cloud->size() / 256 + 1);

    for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
        FECUGB_BlockKey key = get_block_key(cloud->points[i]);
        auto it = block_to_id.find(key);
        int bid;
        if (it == block_to_id.end()) {
            bid = static_cast<int>(blocks.size());
            block_to_id[key] = bid;
            blocks.push_back(FECUGB_BlockData{key});
        } else {
            bid = it->second;
        }
        blocks[bid].owner_indices.push_back(i);
    }

    auto tp1 = Clock::now();
    double partition_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    auto te0 = Clock::now();

    for (int global_idx = 0; global_idx < static_cast<int>(cloud->size()); ++global_idx) {
        const auto& pt = cloud->points[global_idx];

        int bx0 = static_cast<int>(std::floor((pt.x - min_x - tolorance) / block_size));
        int by0 = static_cast<int>(std::floor((pt.y - min_y - tolorance) / block_size));
        int bz0 = static_cast<int>(std::floor((pt.z - min_z - tolorance) / block_size));
        int bx1 = static_cast<int>(std::floor((pt.x - min_x + tolorance) / block_size));
        int by1 = static_cast<int>(std::floor((pt.y - min_y + tolorance) / block_size));
        int bz1 = static_cast<int>(std::floor((pt.z - min_z + tolorance) / block_size));

        for (int bx = bx0; bx <= bx1; ++bx) {
            for (int by = by0; by <= by1; ++by) {
                for (int bz = bz0; bz <= bz1; ++bz) {
                    FECUGB_BlockKey key{bx, by, bz};
                    auto it = block_to_id.find(key);
                    if (it != block_to_id.end()) {
                        blocks[it->second].expanded_indices.push_back(global_idx);
                    }
                }
            }
        }
    }

    for (auto& block : blocks) {
        std::sort(block.expanded_indices.begin(), block.expanded_indices.end());
        block.expanded_indices.erase(
            std::unique(block.expanded_indices.begin(), block.expanded_indices.end()),
            block.expanded_indices.end()
        );
    }

    auto te1 = Clock::now();
    double expand_ms = std::chrono::duration<double, std::milli>(te1 - te0).count();

    std::vector<std::vector<FECUGB_LocalCluster>> block_local_clusters(blocks.size());
    std::vector<FECUGB_LocalStats> block_stats(blocks.size());

    double local_build_max = 0.0;
    double local_search_max = 0.0;
    double local_merge_max = 0.0;
    double local_final_max = 0.0;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bid = 0; bid < static_cast<int>(blocks.size()); ++bid) {
        FECUGB_LocalStats stats;
        block_local_clusters[bid] = FECUGB_LocalClusterOnly(
            cloud,
            blocks[bid].expanded_indices,
            blocks[bid].owner_indices,
            tolorance,
            max_n,
            stats
        );
        block_stats[bid] = stats;
    }

    for (const auto& s : block_stats) {
        local_build_max = std::max(local_build_max, s.build_ms);
        local_search_max = std::max(local_search_max, s.search_ms);
        local_merge_max = std::max(local_merge_max, s.merge_ms);
        local_final_max = std::max(local_final_max, s.final_ms);
    }

    auto tg0 = Clock::now();

    std::vector<FECUGB_LocalCluster> all_local_clusters;
    all_local_clusters.reserve(blocks.size() * 2);

    for (auto& clusters : block_local_clusters) {
        for (auto& c : clusters) {
            all_local_clusters.push_back(std::move(c));
        }
    }

    FECUGB_DisjointSet dsu(static_cast<int>(all_local_clusters.size()));
    std::unordered_map<int, std::vector<int>> point_to_cluster_ids;
    point_to_cluster_ids.reserve(cloud->size());

    for (int cid = 0; cid < static_cast<int>(all_local_clusters.size()); ++cid) {
        for (int gidx : all_local_clusters[cid].all_global_indices) {
            point_to_cluster_ids[gidx].push_back(cid);
        }
    }

    for (auto& kv : point_to_cluster_ids) {
        auto& cluster_ids = kv.second;
        if (cluster_ids.size() <= 1) continue;
        int first = cluster_ids.front();
        for (int i = 1; i < static_cast<int>(cluster_ids.size()); ++i) {
            dsu.unite(first, cluster_ids[i]);
        }
    }

    auto tg1 = Clock::now();
    double global_merge_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();

    auto tf0 = Clock::now();

    std::unordered_map<int, std::vector<int>> root_to_owner_points;
    root_to_owner_points.reserve(all_local_clusters.size());

    for (int cid = 0; cid < static_cast<int>(all_local_clusters.size()); ++cid) {
        int root = dsu.find(cid);
        auto& dst = root_to_owner_points[root];
        const auto& src = all_local_clusters[cid].owner_global_indices;
        dst.insert(dst.end(), src.begin(), src.end());
    }

    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices.reserve(root_to_owner_points.size());

    for (auto& kv : root_to_owner_points) {
        auto& indices = kv.second;
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

        if (static_cast<int>(indices.size()) >= min_component_size) {
            pcl::PointIndices inliers;
            inliers.indices = std::move(indices);
            cluster_indices.push_back(std::move(inliers));
        }
    }

    auto tf1 = Clock::now();
    double output_final_ms = std::chrono::duration<double, std::milli>(tf1 - tf0).count();

    auto t_total_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();

    const double build_ms = partition_ms + expand_ms + local_build_max;
    const double search_ms = local_search_max;
    const double merge_ms = local_merge_max + global_merge_ms;
    const double final_ms = local_final_max + output_final_ms;

    std::cout << std::fixed << std::setprecision(3)
              << "FEC_Union_Grid_Block total= " << total_ms << " ms "
              << "build= " << build_ms << " ms "
              << "search= " << search_ms << " ms "
              << "merge= " << merge_ms << " ms "
              << "final= " << final_ms << " ms "
              << "clusters=" << cluster_indices.size() << "\n";

    return cluster_indices;
}

#endif
