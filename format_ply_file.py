import numpy as np
import open3d as o3d


def load_ply_as_xyz(filepath):
    """
    Đọc file .ply và trả về đám mây điểm dưới dạng numpy array (XYZ).
    
    Parameters:
        filepath (str): Đường dẫn đến file .ply.
        
    Returns:
        np.ndarray: Mảng numpy có dạng (N, 3), mỗi dòng là một điểm (x, y, z).
    """
    point_cloud = o3d.io.read_point_cloud(filepath)
    xyz = np.asarray(point_cloud.points)
    return xyz


def xyz_to_ply(xyz_pointcloud, output_ply_file):
    """
    Chuyển đổi file .txt chứa tọa độ điểm thành file .ply.
    
    Parameters:
        input_txt_file (str): Đường dẫn đến file .txt chứa tọa độ điểm.
        output_ply_file (str): Đường dẫn để lưu file .ply đầu ra.
    """
    # Bước 1: Đọc dữ liệu tọa độ x y z từ file txt
    points = xyz_pointcloud

    # Bước 2: Tạo đối tượng point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Bước 3: Ước lượng vector pháp tuyến (normals)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # (Tuỳ chọn) Căn chỉnh hướng normals cho thống nhất
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # (Tuỳ chọn) Kiểm tra hiển thị normals
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # Bước 4: Lưu ra file .ply có normals
    o3d.io.write_point_cloud(output_ply_file, pcd, write_ascii=True)

    

ply_input_file = "output.ply"
ply_output_file = "output.ply"

xyz_pointcloud = load_ply_as_xyz(ply_input_file)
xyz_to_ply(xyz_pointcloud, ply_output_file)

