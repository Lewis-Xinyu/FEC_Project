import time
import numpy as np
import open3d as o3d
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    min_component_size = 100
    tolerance = 0.5
    
    start = time.perf_counter()
    
    # 1. 读取原汁原味的 112 万点云 (绝对不降维)
    cloud = o3d.io.read_point_cloud("../data/street.ply")
    points = np.asarray(cloud.points)
    size = len(points)
    print(f"📦 读取原始点云数量: {size}")

    if size < min_component_size:
        raise ValueError("Could not find any cluster")

    # 2. 并查集初始化
    parent = np.arange(size)

    def find(i):
        root = i
        while root != parent[root]:
            root = parent[root]
        curr = i
        while curr != root:
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        return root

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # ==========================================
    # 3. 核心替换：构建体素网格空间索引 (Voxel Grid)
    # ==========================================
    print("🕸️ 正在构建体素网格索引 (Spatial Hashing)...")
    voxel_size = tolerance
    min_bound = np.min(points, axis=0)
    
    # 计算每个点所属的体素坐标
    voxel_coords = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    
    # 用哈希表 (字典) 将点索引按体素坐标分类
    voxel_grid = defaultdict(list)
    for idx, coord in enumerate(voxel_coords):
        voxel_grid[tuple(coord)].append(idx)
        
    print(f"📐 空间共被划分为 {len(voxel_grid)} 个非空体素区块。")

    # ==========================================
    # 4. 体素邻域搜索 (代替 KD-Tree)
    # ==========================================
    print("🔗 正在执行体素邻域搜索与并查集合并...")
    
    # 优雅生成 14 个半空间探测方向 (包含自己 1 个 + 周围 13 个)
    offsets = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dz > 0: offsets.append((dx, dy, dz))
                elif dz == 0 and dy > 0: offsets.append((dx, dy, dz))
                elif dz == 0 and dy == 0 and dx >= 0: offsets.append((dx, dy, dz))
                
    tolerance_sq = tolerance ** 2

    # 遍历每一个有点的体素
    for voxel_coord, pt_indices in voxel_grid.items():
        pt_coords = points[pt_indices]
        
        # 探查 14 个方向的邻接体素
        for dx, dy, dz in offsets:
            neighbor_voxel = (voxel_coord[0]+dx, voxel_coord[1]+dy, voxel_coord[2]+dz)
            
            # 如果旁边的体素没有点，直接跳过
            if neighbor_voxel not in voxel_grid:
                continue
                
            neighbor_indices = voxel_grid[neighbor_voxel]
            neighbor_coords = points[neighbor_indices]
            
            # 🔥 Numpy 魔法：矩阵广播一次性算出两个体素内所有点的两两平方距离
            # pt_coords 维度 (N, 3) -> (N, 1, 3)
            # neighbor_coords 维度 (M, 3) -> (1, M, 3)
            # diff 维度 -> (N, M, 3)
            diff = pt_coords[:, np.newaxis, :] - neighbor_coords[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            
            # 找到距离小于 tolerance 的点对的行号和列号
            row_idx, col_idx = np.where(dist_sq <= tolerance_sq)
            
            # 将符合条件的点对用并查集合并
            for r, c in zip(row_idx, col_idx):
                p_i = pt_indices[r]
                p_j = neighbor_indices[c]
                
                # 特殊处理：如果是和自身体素内的点比较，防止自己和自己连，或者重复连
                if dx == 0 and dy == 0 and dz == 0:
                    if p_i < p_j: 
                        union(p_i, p_j)
                else:
                    union(p_i, p_j)

    # ==========================================
    # 5. 提取与渲染
    # ==========================================
    print("📦 正在整理最终的聚类结果...")
    clusters_dict = defaultdict(list)
    for i in range(size):
        root = find(i)
        clusters_dict[root].append(i)

    final_indices = []
    cluster_colors = []
    np.random.seed(42)
    
    valid_clusters = 0
    for root, indices_list in clusters_dict.items():
        if len(indices_list) >= min_component_size:
            valid_clusters += 1
            final_indices.extend(indices_list)
            color = np.random.rand(3)
            cluster_colors.extend([color] * len(indices_list))

    end = time.perf_counter()
    print(f"⏱️ 体素网格搜索+并查集 总耗时: {end - start:.3f} 秒")
    print(f"🎯 共找到有效聚类簇数: {valid_clusters}")

    if len(final_indices) > 0:
        final_cloud = cloud.select_by_index(final_indices)
        final_cloud.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        print("✨ 渲染完成！")
        o3d.visualization.draw_geometries([final_cloud], window_name="Voxel Grid FEC")
    else:
        print("没有找到符合条件的聚类。")

if __name__ == "__main__":
    main()