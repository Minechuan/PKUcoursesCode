# visualize_two_pointclouds.py
import argparse
import numpy as np
import open3d as o3d


def load_txt_pointcloud(path: str) -> np.ndarray:
    """
    读取 N x 3 的 txt 点云文件
    """
    pts = np.loadtxt(path, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 3:
        raise ValueError(f"{path} 不是 N x 3 格式，实际 shape={pts.shape}")
    return pts


def np_to_o3d_pcd(points: np.ndarray, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)  # RGB in [0,1]
    return pcd


def main():
    parser = argparse.ArgumentParser(description="同时可视化两个 txt 点云")
    parser.add_argument("--pcd1", default="D:\\Courses\\sixth_semester\\ComputerVision\\labs\\lab3\\results\\pc_from_depth.txt", type=str, help="第一个点云 txt 路径")
    parser.add_argument("--pcd2", default="D:\\Courses\\sixth_semester\\ComputerVision\\labs\\lab3\\results\\pc_from_depth.txt", type=str, help="第二个点云 txt 路径")
    args = parser.parse_args()

    pts1 = load_txt_pointcloud(args.pcd1)
    pts2 = load_txt_pointcloud(args.pcd2)

    pcd1 = np_to_o3d_pcd(pts1, [1.0, 0.2, 0.2])  # 红色
    pcd2 = np_to_o3d_pcd(pts2, [0.2, 0.6, 1.0])  # 蓝色

    o3d.visualization.draw_geometries(
        [pcd1],
        window_name="Two Point Clouds",
        width=1280,
        height=720
    )


if __name__ == "__main__":
    main()
