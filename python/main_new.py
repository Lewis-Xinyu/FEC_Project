import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    min_component_size = 100
    tolerance = 0.5 # 恢复你原来的容差

    start = time.perf_counter()
    
    # 1. 读取点云
    cloud = o3d.io.read_point_cloud("../data/street.ply")
    print(f"原始点云数量: {len(cloud.points)}")

    # ==========================================
    # 👑 绝对不能省的一步：体素降采样 (Voxel Downsampling)
    # 把空间切成 20cm 的小网格，每个网格只留 1 个点
    # ==========================================
    print("正在进行体素降维压缩...")
    cloud = cloud.voxel_down_sample(voxel_size=0.2)
    
    points = np.asarray(cloud.points)
    size = len(points)
    print(f"降维后点云数量: {size} (计算量将骤降上千倍！)")

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

    # 3. KD-Tree 批量查询
    print("正在构建 KD-Tree 并查找近邻点对...")
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=tolerance)
    print(f"共找到 {len(pairs)} 对近邻点！")

    print("正在执行并查集合并...")
    for i, j in pairs:
        union(i, j)

    # 4. 提取聚类结果
    print("正在整理聚类结果...")
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for i in range(size):
        root = find(i)
        clusters_dict[root].append(i)

    # 5. 过滤噪点，分配颜色
    final_indices = []
    cluster_colors = []
    np.random.seed(42)
    
    for root, indices in clusters_dict.items():
        if len(indices) >= min_component_size:
            final_indices.extend(indices)
            color = np.random.rand(3)
            cluster_colors.extend([color] * len(indices))

    end = time.perf_counter()
    print(f"算法总耗时: {end - start:.3f} 秒")

    # 6. 使用 Open3D 渲染
    if len(final_indices) > 0:
        final_cloud = cloud.select_by_index(final_indices)
        final_cloud.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        print("准备弹出 3D 窗口...")
        o3d.visualization.draw_geometries([final_cloud], window_name="Fast Euclidean Clustering")
    else:
        print("没有找到符合条件的聚类。")

if __name__ == "__main__":
    main()