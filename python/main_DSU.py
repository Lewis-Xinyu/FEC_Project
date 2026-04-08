import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # ==========================================
    # 完全按照原版的参数，坚决不进行体素降维！
    # ==========================================
    min_component_size = 100
    tolerance = 0.5
    max_n = 50  # 还原原版代码中的最大邻居数限制（防止内存爆炸的关键）

    start = time.perf_counter()
    
    # 1. 读取完整点云（112万点原汁原味）
    cloud = o3d.io.read_point_cloud("../data/street.ply")
    points = np.asarray(cloud.points)
    size = len(points)
    print(f"✅ 严格复现模式：读取原始点云数量: {size}")

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
    # 3. 核心替换：带上限的 KD-Tree 并发查询
    # ==========================================
    print("🔍 正在构建 KD-Tree 并进行带上限的多线程近邻搜索...")
    tree = cKDTree(points)
    
    # workers=-1 表示火力全开，调用 Mac 的所有 CPU 核心并发搜索！
    distances, indices = tree.query(points, k=max_n, distance_upper_bound=tolerance, workers=-1)

    print("🔗 正在执行并查集合并 (纯计算，约需 10~20 秒，请稍候)...")
    # 遍历每个点和它的（最多50个）邻居进行合并
    for i in range(size):
        for j in indices[i]:
            # cKDTree 会把没找到邻居的空位用 size 填充，所以用 j < size 排除无效索引
            if j < size:  
                union(i, j)

    # 4. 提取聚类结果
    print("📦 正在整理最终的聚类结果...")
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for i in range(size):
        root = find(i)
        clusters_dict[root].append(i)

    # 5. 过滤噪点，分配颜色
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
    print(f" 算法总耗时: {end - start:.3f} 秒")
    print(f" 共找到有效聚类簇数: {valid_clusters}")

    # ==========================================
    # 6. 保留 Open3D 高性能渲染 (避免段错误)
    # ==========================================
    if len(final_indices) > 0:
        final_cloud = cloud.select_by_index(final_indices)
        final_cloud.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        print("✨ 准备弹出 3D 窗口...")
        o3d.visualization.draw_geometries([final_cloud], window_name="Original Scale FEC")
    else:
        print("没有找到符合条件的聚类。")

if __name__ == "__main__":
    main()