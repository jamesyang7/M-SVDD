import matplotlib
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

matplotlib.use('TKAgg')

def read_binary_point_cloud(bin_file):
    # Assuming XYZI coordinates (float32) for each point
    point_cloud = np.fromfile(bin_file, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 4)
    point_cloud = point_cloud[:, :3]

    return point_cloud

# def extract_and_downsample_point_cloud(input_file, region_of_interest, num_points=2048):
#     point_cloud = read_binary_point_cloud(input_file)
#     # Define the region of interest (ROI) using a bounding box
#     mask = np.all(
#         (point_cloud >= region_of_interest[0]) & (point_cloud <= region_of_interest[1]), axis=1
#     )
#     # Apply the mask to extract points within the region of interest
#     roi_point_cloud = point_cloud[mask]
#     # Shuffle the points randomly
#     np.random.shuffle(roi_point_cloud)
#     # Select the desired number of points for downsampling
#     downsampled_point_cloud = roi_point_cloud[:num_points]
#     # print(downsampled_point_cloud.shape)
#     return downsampled_point_cloud


def extract_and_downsample_point_cloud(input_file, region_of_interest, num_points=2048, retrain_selected_region=False):
    point_cloud = read_binary_point_cloud(input_file)

    # Define the region of interest (ROI) using a bounding box
    mask = np.all(
        (point_cloud >= region_of_interest[0]) & (point_cloud <= region_of_interest[1]), axis=1
    )

    # Apply the mask to extract points within the region of interest
    roi_point_cloud = point_cloud[mask]

    # Retrain only the selected region if specified
    if retrain_selected_region:
        # Modify this part to include your retraining logic for the selected region
        # For example, you can replace the following line with your retraining code:
        roi_point_cloud = retrain_selected_points(roi_point_cloud)

    # Check if the number of points is less than the specified num_points
    if len(roi_point_cloud) < num_points:
        # Repeat or replicate points to reach the desired number of points
        replicated_points = replicate_points(roi_point_cloud, num_points)
        return replicated_points

    # Shuffle the points randomly
    np.random.shuffle(roi_point_cloud)

    # Select the desired number of points for downsampling
    downsampled_point_cloud = roi_point_cloud[:num_points]

    # print(downsampled_point_cloud.shape)
    return downsampled_point_cloud


# Example retraining function (replace with your actual retraining logic)
def retrain_selected_points(selected_points):
    # Your retraining logic goes here
    # For example, you can use a machine learning model to retrain the selected points
    # Replace the following line with your actual retraining code:
    retrained_points = selected_points * 2  # This is just a placeholder, replace with your actual code

    return retrained_points


def replicate_points(points, target_num_points):
    # Repeat or replicate points to reach the desired number of points
    num_repeats = int(np.ceil(target_num_points / len(points)))
    replicated_points = np.tile(points, (num_repeats, 1))[:target_num_points, :]

    return replicated_points

def visualize_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

def visualize_gen(points, bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
                           azimuth=30, elevation=30, distance=3):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set point cloud display range
    mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    displayed_points = points[mask]

    ax.scatter(displayed_points[:, 0], displayed_points[:, 1], displayed_points[:, 2], s=20, c='b', marker='o')

    # Set view parameters
    ax.view_init(azim=azimuth, elev=elevation)
    ax.dist = distance

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def visualize_compare(points1, points2, bbox_min=(0.1, -2 ,-0.8), bbox_max=(2, 2, 0.8),
                            azimuth=30, elevation=30, distance=3):
    fig = plt.figure(figsize=(12, 6))

    # Plot for Point Cloud 1
    ax1 = fig.add_subplot(121, projection='3d')
    mask1 = np.all((points1 >= bbox_min) & (points1 <= bbox_max), axis=1)
    displayed_points1 = points1[mask1]
    ax1.scatter(displayed_points1[:, 0], displayed_points1[:, 1], displayed_points1[:, 2], s=20, c='b', marker='o')
    ax1.set_title('Point Cloud 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(azim=azimuth, elev=elevation)
    ax1.dist = distance
    ax1.set_xlim(bbox_min[0], bbox_max[0])
    ax1.set_ylim(bbox_min[1], bbox_max[1])
    ax1.set_zlim(bbox_min[2], bbox_max[2])

    # Plot for Point Cloud 2
    ax2 = fig.add_subplot(122, projection='3d')
    mask2 = np.all((points2 >= bbox_min) & (points2 <= bbox_max), axis=1)
    displayed_points2 = points2[mask2]
    ax2.scatter(displayed_points2[:, 0], displayed_points2[:, 1], displayed_points2[:, 2], s=20, c='r', marker='^')
    ax2.set_title('Point Cloud 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(azim=azimuth, elev=elevation)
    ax2.dist = distance
    ax2.set_xlim(bbox_min[0], bbox_max[0])
    ax2.set_ylim(bbox_min[1], bbox_max[1])
    ax2.set_zlim(bbox_min[2], bbox_max[2])

    plt.tight_layout()
    plt.show()


def update(frame, point_pairs, ax1, ax2, bbox_min, bbox_max, azimuth, elevation, distance):
    ax1.cla()
    ax2.cla()

    # Plot for Point Cloud 1
    points1 = point_pairs[0][frame]
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], s=20, c='b', marker='o')
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(azim=azimuth, elev=elevation)
    ax1.dist = distance
    ax1.set_xlim(bbox_min[0], bbox_max[0])
    ax1.set_ylim(bbox_min[1], bbox_max[1])
    ax1.set_zlim(bbox_min[2], bbox_max[2])

    # Plot for Point Cloud 2
    points2 = point_pairs[1][frame]
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], s=20, c='r', marker='^')
    ax2.set_title('Generated')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(azim=azimuth, elev=elevation)
    ax2.dist = distance
    ax2.set_xlim(bbox_min[0], bbox_max[0])
    ax2.set_ylim(bbox_min[1], bbox_max[1])
    ax2.set_zlim(bbox_min[2], bbox_max[2])

def visualize_compare_animation(point_pairs, bbox_min=(0.01, -1 ,-0.5), bbox_max= (5, 1, 1),
                                 azimuth=30, elevation=30, distance=3):
    fig = plt.figure(figsize=(12, 6))

    # Plot for Point Cloud 1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(azim=azimuth, elev=elevation)
    ax1.dist = distance
    ax1.set_xlim(bbox_min[0], bbox_max[0])
    ax1.set_ylim(bbox_min[1], bbox_max[1])
    ax1.set_zlim(bbox_min[2], bbox_max[2])

    # Plot for Point Cloud 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Generated')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(azim=azimuth, elev=elevation)
    ax2.dist = distance
    ax2.set_xlim(bbox_min[0], bbox_max[0])
    ax2.set_ylim(bbox_min[1], bbox_max[1])
    ax2.set_zlim(bbox_min[2], bbox_max[2])

    ani = FuncAnimation(fig, update, frames=len(point_pairs[0]), fargs=(point_pairs, ax1, ax2, bbox_min, bbox_max, azimuth, elevation, distance), interval=200)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you receive points one pair at a time from a generator or iterator
# visualize_compare_animation(point_pairs_generator)




# def visualize_gen(points, bbox_min=(-2, -2, -2), bbox_max=(2, 2, 2),
#                            azimuth=90, elevation=90, distance=1000,
#                            window_name='PointCloud', width=800, height=600):
#     # 创建点云对象
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#
#     # 设置点云的显示范围
#     bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
#     point_cloud = point_cloud.crop(bbox)
#
#     # 创建可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_name, width=width, height=height)
#
#     # 添加点云到窗口
#     vis.add_geometry(point_cloud)
#
#     # 获取渲染控制
#     render_option = vis.get_render_option()
#     render_option.background_color = np.asarray([0, 0, 0])  # 设置背景颜色为黑色
#
#     # 获取默认视角控制
#     vis.get_view_control().rotate(azimuth, elevation)
#     vis.get_view_control().translate(0, 0, -distance)  # 设置平移，使点云位于视图中心
#
#     # 运行可视化
#     vis.run()
#     # 关闭窗口
#     vis.destroy_window()

#### example of usage ###
if __name__ == "__main__":
    # Replace 'input_file.bin' and the region_of_interest values with your actual values
    input_file = '/home/kemove/yyz/audio_pointcloud/Data/pointcloud/3/405.bin'
    region_of_interest= [(0.01, -1 ,-0.5), (5, 1, 1)]
    # Extract and downsample the point cloud
    # original_point_cloud, downsampled_point_cloud = extract_and_downsample_point_cloud(input_file, region_of_interest,512)
    downsampled_point_cloud = extract_and_downsample_point_cloud(input_file, region_of_interest,512)
    # Visualize the original and downsampled point clouds
    # visualize_point_cloud(original_point_cloud)
    visualize_compare(downsampled_point_cloud,downsampled_point_cloud,(0.01, -1 ,-0.5), (5, 1, 1))
    # visualize_compare(original_point_cloud,downsampled_point_cloud)
    print(downsampled_point_cloud[:,0])
    print(downsampled_point_cloud[:,1])
# #### example of usage ###



