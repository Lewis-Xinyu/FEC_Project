import time
import numpy as np
import open3d as o3d
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # 完全保留原版的参数
    min_component_size = 100
    tolerance = 0.5
    max_n = 50

    start = time.perf_counter()
    
    # 读取原始 112 万点云
    cloud = o3d.io.read_point_cloud("../data/street.ply")
    cloud_tree = o3d.geometry.KDTreeFlann(cloud)
    points = np.asarray(cloud.points)
    size = len(points)
    
    print(f"✅ 论文复现 Baseline 模式：载入点云 {size} 个")
    if size < min_component_size:
        raise ValueError("Could not find any cluster")

    # 原版的标记数组 (为了稍微防止内存溢出，这里用了 numpy 数组，但不影响算法逻辑)
    marked_indices = np.zeros(size, dtype=int)
    
    tag_num = 1
    temp_tag_num = -1
    
    print("⏳ 正在执行原版暴力扫描聚类 (预计需要 60~80 秒)...")
    
    # ==========================================
    # 核心算法 100% 还原你提供的原始代码逻辑
    # ==========================================
    for i in range(size):
        # 打印进度条，防止你觉得它死机了
        if i % 100000 == 0 and i > 0:
            print(f"   已处理 {i} 个点...")
            
        if marked_indices[i] == 0:
            # 限制最多找 50 个邻居
            [k, idx, _] = cloud_tree.search_hybrid_vector_3d(cloud.points[i], tolerance, max_n)
            
            min_tag_num = tag_num
            for j in idx:
                if (marked_indices[j] > 0) and (marked_indices[j] < min_tag_num):
                    min_tag_num = marked_indices[j]
                    
            for j in idx:
                temp_tag_num = marked_indices[j]
                if temp_tag_num > min_tag_num:
                    # ⚠️ 这里就是传说中的 O(N^2) 性能黑洞！
                    # 每次发现冲突，就要遍历 112 万次！
                    for k_idx in range(size):
                        if marked_indices[k_idx] == temp_tag_num:
                            marked_indices[k_idx] = min_tag_num
                marked_indices[j] = min_tag_num
                
        tag_num += 1

    end = time.perf_counter()
    print(f"⏱️ 原版算法总耗时: {end - start:.3f} 秒")

    # ==========================================
    # 渲染部分：替换为 Open3D，防止 Matplotlib 崩溃
    # ==========================================
    print("📦 正在提取有效聚类并准备画图...")
    
    # 统计每个 tag 包含的点数
    unique_tags, counts = np.unique(marked_indices, return_counts=True)
    
    final_indices = []
    cluster_colors = []
    np.random.seed(42)
    
    valid_clusters = 0
    # 过滤掉噪点 (点数小于 min_component_size 且 tag_num 为 0 的)
    for tag, count in zip(unique_tags, counts):
        if tag > 0 and count >= min_component_size:
            valid_clusters += 1
            # 找到属于这个 tag 的所有点的索引
            idx_list = np.where(marked_indices == tag)[0]
            final_indices.extend(idx_list)
            # 随机分配颜色
            color = np.random.rand(3)
            cluster_colors.extend([color] * len(idx_list))

    print(f"🎯 共找到有效聚类簇数: {valid_clusters}")

    if len(final_indices) > 0:
        final_cloud = cloud.select_by_index(final_indices)
        final_cloud.colors = o3d.utility.Vector3dVector(np.array(cluster_colors))
        print("✨ 渲染完成！")
        o3d.visualization.draw_geometries([final_cloud], window_name="Original Slow FEC")
    else:
        print("没有找到符合条件的聚类。")

if __name__ == "__main__":
    main()