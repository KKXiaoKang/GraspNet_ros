""" Demo to show prediction results.
    Author: KKXiaoKang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import rospy
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image as PILImage

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import cameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

print( " cfgs : ", cfgs)
data_dir = 'doc/example_data'

class GraspNetNode:

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('graspnet_node', anonymous=True)

        # Initialize 
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        
        # 模型路径
        self.model_path = "/home/lab/GenDexGrasp/GraspNet_ros/STL/Makerbot_Gripper_Assembled.stl"

        # 预定义掩膜，表示感兴趣的工作区域
        self.workspace_mask = np.array(PILImage.open(os.path.join(data_dir, 'workspace_mask.png')))

        # ROS Subscribers
        rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)

        # ROS Publishers
        self.marker_pub = rospy.Publisher('/graspnet_visualization_marker_array', MarkerArray, queue_size=10)

        # Load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.get_net()

    def get_net(self):
        """
            初始化Net网络
        """
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        
        # Load checkpoint
        checkpoint = torch.load(cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
        
        # set model to eval mode
        net.to(self.device)
        net.eval()
        return net
    
    def rgb_callback(self, msg):
        """Callback for RGB image."""
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        # rospy.loginfo("Received RGB image.")

    def depth_callback(self, msg):
        """Callback for Depth image."""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        # rospy.loginfo("Received depth image.")

    def camera_info_callback(self, msg):
        """Callback for CameraInfo."""
        self.camera_info = msg
        # rospy.loginfo("Received camera info.")

    def process_data(self):
        global data_dir
        """Process images and generate point cloud."""
        rospy.loginfo("Processing data...")
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return None
        # RGB 归一化
        rgb_image_normalized = np.array(self.rgb_image, dtype=np.float32) / 255.0

        # Extract camera parameters
        camera_info = cameraInfo(
            self.camera_info.width,
            self.camera_info.height,
            self.camera_info.K[0], self.camera_info.K[4],  # fx, fy
            self.camera_info.K[2], self.camera_info.K[5],  # cx, cy
            np.array([[1000.]]),  # self.camera_info.D[0]  # depth scale
        )

        # print("camera_info :")
        # print("  width:", camera_info.width)
        # print("  height:", camera_info.height)
        # print("  fx:", camera_info.fx)
        # print("  fy:", camera_info.fy)
        # print("  cx:", camera_info.cx)
        # print("  cy:", camera_info.cy)
        # print("  scale:", camera_info.scale)
        # print("  scale type:", type(camera_info.scale))

        # create point cloud
        cloud = create_point_cloud_from_depth_image(self.depth_image, camera_info, organized=True)
        cloud_sampled = torch.from_numpy(cloud[np.newaxis].astype(np.float32)).to(self.device)

        # get valid points
        """
            获取感兴趣的工作区域 | 过滤无效点 
        """
        mask = (self.workspace_mask & (self.depth_image > 0))
        cloud_masked = cloud[mask]
        color_masked = rgb_image_normalized[mask]

        # sample points
        """
            采样点云
        """
        if len(cloud_masked) >= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        """
            转换数据格式 | 转换为open3D的点云格式
        """
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud
    
    def get_grasps(self, end_points):
        """Run inference to get grasp poses."""
        rospy.loginfo("Running inference...")
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)

        return gg
    

    def collision_detection(self, gg, cloud):
        """
            碰撞检测 | NMS输出
        """
        rospy.loginfo("Collision detection...")
        # 碰撞检测
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

        # NMS | 排序分数 | 取前50个抓取姿态 | 是否进行open3d可视化
        gg.nms()
        gg.sort_by_score()
        # gg = gg[:50] # 取前50个抓取姿态
        gg = gg[:5] # 取前5个抓取姿态
        # grippers = gg.to_open3d_geometry_list()
        # o3d.visualization.draw_geometries([cloud, *grippers])
        return gg

    def publish_grasps(self, grasps, frame_id='camera_color_optical_frame'):
        """Publish grasp poses as markers."""
        rospy.loginfo("Publishing grasps...")
        marker_array = MarkerArray()
        for i, grasp in enumerate(grasps):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.mesh_resource = "file://" + self.model_path

            marker.pose = self.grasp_pose_to_camera_pose(grasp) # Convert grasp to ROS pose
            
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.id = i
            
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def grasp_pose_to_camera_pose(self, grasp_pose):
        """Transform grasp pose to camera pose."""
        # TODO: implement this function
        pass
    
    def run(self):
        """Main loop to process and publish grasps."""
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.rgb_image is not None and self.depth_image is not None:
                # 处理为open3d格式
                end_points, cloud = self.process_data()
                # 获取抓取姿态
                grasps = self.get_grasps(end_points)
                # 碰撞检测
                if cfgs.collision_thresh > 0:
                    grasps = self.collision_detection(grasps, np.array(cloud.points))
                # 发布最终抓取姿态
                print( " grasps : ", grasps)

                # self.publish_grasps(grasps, frame_id='camera_color_optical_frame')
            rate.sleep()

if __name__ == '__main__':
    node = GraspNetNode()
    node.run()