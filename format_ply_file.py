import numpy as np
import open3d as o3d

class PointCloudProcessing:
    def __init__(self, filepath):
        self.pcd = o3d.io.read_point_cloud(filepath)

    def load_ply_as_xyz(self, filepath):
        """
        Đọc file .ply và trả về đám mây điểm dưới dạng numpy array (XYZ).
        """
        self.pcd = o3d.io.read_point_cloud(filepath)
        xyz = np.asarray(self.pcd.points)
        return xyz

    def xyz_to_ply(self, xyz_pointcloud, output_ply_file):
        """
        Chuyển đổi đám mây điểm từ numpy array (XYZ) thành file .ply.
        """
        points = xyz_pointcloud

        # Tạo đối tượng point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Ước lượng vector pháp tuyến
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        # Căn chỉnh hướng pháp tuyến
        pcd.orient_normals_consistent_tangent_plane(k=30)

        # Lưu ra file .ply có pháp tuyến
        o3d.io.write_point_cloud(output_ply_file, pcd, write_ascii=True)

    def filter_with_dbscan(self, eps=0.13, min_points=3):
        """
        Lọc nhiễu sử dụng thuật toán DBSCAN.
        """
        print("Filtering outliers using DBSCAN...")
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        if labels.max() < 0:
            print("No clusters found.")
            return

        # Giữ lại cụm có số lượng điểm lớn nhất
        largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
        indices = np.where(labels == largest_cluster)[0]
        self.pcd = self.pcd.select_by_index(indices)
        print("Filtered point cloud.")

    def remove_outliers(self):
        """
        Lọc nhiễu bằng phương pháp thống kê (Statistical Outlier Removal).
        """
        print("Removing noise using Statistical Outlier Removal...")
        cl_stat, ind_stat = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.display_inlier_outlier(self.pcd, ind_stat)
        self.pcd = self.pcd.select_by_index(ind_stat)

    def display_inlier_outlier(self, pcd, ind_stat):
        """
        Hiển thị các điểm inlier và outlier.
        """
        inlier_cloud = pcd.select_by_index(ind_stat)
        outlier_cloud = pcd.select_by_index(ind_stat, invert=True)
        print("Displaying inliers and outliers...")
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Inliers and Outliers")


# Đọc và xử lý đám mây điểm từ file .ply
ply_input_file = "output.ply"
ply_output_file = "output_filtered.ply"

pcd_processor = PointCloudProcessing(ply_input_file)
xyz_pointcloud = pcd_processor.load_ply_as_xyz(ply_input_file)

# Lọc nhiễu bằng DBSCAN và SOR
pcd_processor.filter_with_dbscan(eps=0.13, min_points=3)
pcd_processor.remove_outliers()

# Chuyển đổi lại đám mây điểm sau khi xử lý sang file .ply
pcd_processor.xyz_to_ply(xyz_pointcloud, ply_output_file)
