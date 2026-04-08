import time
import numpy as np
import open3d as o3d
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # 严格对齐原版参数：不降维，限制 50 个邻居
    min_component_size = 100
    tolerance = 0.5
    max_n = 50

    start = time.perf_counter()
    
    # 1. 读取原版点云
    cloud = o3d.io.read_point_cloud("../data/street.ply")
    points = np.asarray(cloud.points)
    size = len(points)
    print(f"✅ 消融实验模式：读取原始点云数量: {size} (使用原版 KDTreeFlann)")

    if size < min_component_size:
        raise ValueError("Could not find any cluster")

    # 2. 并查集初始化 (替代原版的 marked_indices 全局替换)
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
    # 3. 原版近邻搜索：使用 Open3D 自带的普通 KD-Tree
    # ==========================================
    print("🌲 正在构建 Open3D 普通 KD-Tree...")
    cloud_tree = o3d.geometry.KDTreeFlann(cloud)

    print("⏳ 正在执行原版逐点扫描与并查集合并 (预计耗时 30~50 秒)...")
    
    # ⚠️ 性能瓶颈在这里：112 万次 Python 解释器向 C++ 库发起的函数调用
    for i in range(size):
        if i % 200000 == 0 and i > 0:
            print(f"   已扫描 {i} 个点...")
            
        # 完全采用原版的搜索 API
        [k, idx, _] = cloud_tree.search_hybrid_vector_3d(cloud.points[i], tolerance, max_n)
        
        # 使用并查集进行合并 (彻底砍掉了原版 O(N^2) 的 for 循环)
        for j in idx:
            if i != j: # 防止自己和自己连
                union(i, j)

    # ==========================================
    # 4. 提取与渲染
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
    print(f"⏱️ 普通 KD-Tree + 并查集 总耗时: {end - start:.3f} 秒")
    print(f"🎯 共找到有效聚类簇数: {valid_clusters}")

    if len(final_indices) > 0:
        final_cloud = cloud.select_by_index(final_indices)
        final_cloud.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        print("✨ 渲染完成！")
        o3d.visualization.draw_geometries([final_cloud], window_name="Ablation: Normal KDTree + Union Find")
    else:
        print("没有找到符合条件的聚类。")

if __name__ == "__main__":
    main()