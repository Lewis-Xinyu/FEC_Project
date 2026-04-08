import time
import numpy as np
import open3d as o3d
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # 严格控制变量的参数
    min_component_size = 100
    tolerance = 0.5
    max_n = 50  # 👑 严格对齐原版代码：最多只找50个邻居！
    
    start = time.perf_counter()
    
    # 1. 读取原汁原味的 112 万点云
    cloud = o3d.io.read_point_cloud("../data/street.ply")
    points = np.asarray(cloud.points)
    size = len(points)
    print(f"📦 学术严谨模式：读取原始点云数量: {size} (不降维)")

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
    # 3. 构建体素网格 (空间哈希)
    # ==========================================
    print("🕸️ 正在构建体素网格索引 (Spatial Hashing)...")
    voxel_size = tolerance
    min_bound = np.min(points, axis=0)
    
    voxel_coords = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    
    voxel_grid = defaultdict(list)
    for idx, coord in enumerate(voxel_coords):
        voxel_grid[tuple(coord)].append(idx)

    # ==========================================
    # 4. 体素邻域搜索 (严控 max_n = 50)
    # ==========================================
    print(f"🔗 正在执行体素搜索 (严格限制每个点最多 {max_n} 个近邻)...")
    
    # KNN是不对称的，必须老老实实探查完整的 3x3x3 = 27 个方向
    offsets = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]]
    tolerance_sq = tolerance ** 2

    for voxel_coord, pt_indices in voxel_grid.items():
        # 把属于这 27 个相邻体素里的所有候选点索引收集起来
        candidate_indices = []
        for dx, dy, dz in offsets:
            neighbor_voxel = (voxel_coord[0]+dx, voxel_coord[1]+dy, voxel_coord[2]+dz)
            if neighbor_voxel in voxel_grid:
                candidate_indices.extend(voxel_grid[neighbor_voxel])
                
        candidate_indices = np.array(candidate_indices, dtype=np.int32)
        candidate_coords = points[candidate_indices]
        pt_coords = points[pt_indices]
        
        # 矩阵广播：一次性计算该体素内所有点，到 27 个体素内所有候选点的平方距离
        # 结果维度: (该体素内的点数, 候选点总数)
        diff = pt_coords[:, np.newaxis, :] - candidate_coords[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        
        # 为该体素内的每一个点，进行精确的 KNN 截断
        for i in range(len(pt_indices)):
            p_i = pt_indices[i]
            
            # 找到距离在 tolerance 之内的候选点
            valid_mask = dist_sq[i] <= tolerance_sq
            valid_candidate_idx = candidate_indices[valid_mask]
            valid_dists = dist_sq[i][valid_mask]
            
            # 👑 核心截断逻辑：如果超过了 max_n，就按距离排序，只取前 50 个！
            if len(valid_candidate_idx) > max_n:
                # argsort 返回从小到大排序的索引，我们取前 max_n 个
                sort_idx = np.argsort(valid_dists)[:max_n]
                valid_candidate_idx = valid_candidate_idx[sort_idx]
                
            # 执行并查集合并
            for p_j in valid_candidate_idx:
                if p_i != p_j:  # 防止自己和自己合并
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
    print(f"⏱️ 严格对照版 (网格+排序+并查集) 总耗时: {end - start:.3f} 秒")
    print(f"🎯 共找到有效聚类簇数: {valid_clusters}")

    if len(final_indices) > 0:
        final_cloud = cloud.select_by_index(final_indices)
        final_cloud.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        print("✨ 渲染完成！")
        o3d.visualization.draw_geometries([final_cloud], window_name="Strict Baseline: Voxel Grid + Union Find")
    else:
        print("没有找到符合条件的聚类。")

if __name__ == "__main__":
    main()