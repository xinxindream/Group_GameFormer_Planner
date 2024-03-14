import numpy as np
import math
import argparse
import os
import sys
import pdb
import torch
from scipy.spatial.distance import cdist
from collections import deque

import rosbag
import rospy
# from message_filters import TimeSynchronizer, Subscriber
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, Vector3
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
# from kxdun_localization_msgs.msg import Localization
# from kxdun_perception_msgs.msg import PerceptionObstacle, PerceptionObstacleArray, PerceptionLaneArray, PerceptionLane
from sensor_msgs.msg import Image

# from kxdun_control_msgs.msg import PlanningADCTrajectory, TrajectoryPoint

from Planner.planner import Planner
from run_nuplan_ros_utils import *

class gameformer_planner:
    def __init__(self, bag_path, model_path, device=None):
        self._bag_path = bag_path
        self._model_path = model_path
        self._device = device
        self._num_agents = 20
        self._map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
        self._radius = 60  # [m] query radius scope relative to the current pose.
        self._sample_interval = 0.2
        self._interpolation_method = 'linear'
        self._past_time_horizon = 2  # [seconds]
        self._num_past_poses = 10 * self._past_time_horizon
        self._future_time_horizon = 8  # [seconds]
        self._num_future_poses = 10 * self._future_time_horizon
        self._drop_start_frames = 0
        self._drop_end_frames = 1

        self._frame_id = 'map'

        self._perception_flag = False
        self._ego_flag = False
        self._lanes_flag = False
        self._routes_flag = False
        self._loc_connt = 0

        self._agents_past_queue = deque(maxlen = 22)
        self._ego_past_queue = deque(maxlen = 22)
        self._ego_future_queue = deque(maxlen=self._num_future_poses)

        self._marker_msg = None
        self._img_msg = None
        self._ego_pose_msg = None

        self._per_flag = False
        self._loc_flag = False
        self._img_flag = False

        # self._kxdun_perception_sub = rospy.Subscriber('/kxdun/perception/obstacles', PerceptionObstacleArray, self.perception_callback)
        # self._kxdun_localization_sub = rospy.Subscriber('/kxdun/ego_vehicle/localization', Localization, self.localization_callback)
        # self._kxdun_img_sub = rospy.Subscriber('/zkhy_stereo/left/color', Image, self.img_callback)

        self._original_route_lane_data_x_path = "/data/datasets/xiaoba/lane_center_line/original_route_lane_data_x.npz"
        self._original_route_lane_data_y_path = "/data/datasets/xiaoba/lane_center_line/original_route_lane_data_y.npz"
        self._shift_route_lane_data_x_path = "/data/datasets/xiaoba/lane_center_line/shift_route_lane_data_x.npz"
        self._shift_route_lane_data_y_path = "/data/datasets/xiaoba/lane_center_line/shift_route_lane_data_y.npz"

        self._ego_markers_pub = rospy.Publisher('/ego_car_markers', MarkerArray, queue_size=10)
        self._obj_markers_pub = rospy.Publisher('/objects_perception_markers', MarkerArray, queue_size=10)
        self._img_pub = rospy.Publisher('/cur_img_pub', Image, queue_size=10)
        self._centerline_path_pub_0 = rospy.Publisher('/centerline_path_pub_0', Path, queue_size=10)
        self._centerline_path_pub_1 = rospy.Publisher('/centerline_path_pub_1', Path, queue_size=10)
        self._planner_path_pub = rospy.Publisher('/learn_based_planner_path_pub', Path, queue_size=10)
        self._future_gt_path_pub = rospy.Publisher('/future_gt_path_pub', Path, queue_size=10)

        original_route_lane_data_x = np.load(self._original_route_lane_data_x_path)
        original_route_lane_data_y = np.load(self._original_route_lane_data_y_path)
        shift_route_lane_data_x = np.load(self._shift_route_lane_data_x_path)
        shift_route_lane_data_y = np.load(self._shift_route_lane_data_y_path)
        original_route_lane_data_xy = np.stack((original_route_lane_data_x, original_route_lane_data_y)).transpose()
        shift_route_lane_data_xy = np.stack((shift_route_lane_data_x, shift_route_lane_data_y)).transpose()

        original_route_lane_data_x = np.load(self._original_route_lane_data_x_path)
        original_route_lane_data_y = np.load(self._original_route_lane_data_y_path)
        shift_route_lane_data_x = np.load(self._shift_route_lane_data_x_path)
        shift_route_lane_data_y = np.load(self._shift_route_lane_data_y_path)

        self._original_route_lane_data_xy = np.stack((original_route_lane_data_x, original_route_lane_data_y)).transpose()
        self._shift_route_lane_data_xy = np.stack((shift_route_lane_data_x, shift_route_lane_data_y)).transpose()

        self._planner = Planner(self._model_path, device)
        self._planner._initialize_model()

        self._path0 = Path()
        self._path1 = Path()
        self._future_path = Path()

    def relative_to_absolute_poses(self,origin_pose, relative_poses):

        def matrix_from_pose(pose):
            """
                Converts a 2D pose to a 3x3 transformation matrix

                :param pose: 2D pose (x, y, yaw)
                :return: 3x3 transformation matrix
                """
            return np.array(
                [
                    [np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                    [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
                    [0, 0, 1],
                ]
            )

        def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]):
            """
            Converts a 3x3 transformation matrix to a 2D pose
            :param transform_matrix: 3x3 transformation matrix
            :return: 2D pose (x, y, yaw)
            """
            if transform_matrix.shape != (3, 3):
                raise RuntimeError(f"Expected a 3x3 transformation matrix, got {transform_matrix.shape}")

            heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

            return [transform_matrix[0, 2], transform_matrix[1, 2], heading]

        relative_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(relative_poses[i,:]) for i in range(relative_poses.shape[0])])
        origin_transform = matrix_from_pose(origin_pose)
        absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
        absolute_poses = [pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms]

        return absolute_poses

    def run(self):
        print(self._bag_path)
        bag = rosbag.Bag(self._bag_path)

        agents_list = []
        ego_poses_list = []
        ego_time_list = []
        img_color_list = []
        markers_list  =[]
        stamp_list = []

        agent_topic = '/kxdun/perception/obstacles'
        ego_pose_topic = '/kxdun/ego_vehicle/localization'
        img_color_topic = '/zkhy_stereo/left/color'
        markers_topic = '/objects_markers'
        ego_pose_flag, agent_flag, img_flag = False, False, False

        for i in range(self._original_route_lane_data_xy.shape[0]):
            pose = PoseStamped()
            pose.pose.position.x = self._original_route_lane_data_xy[i,0]
            pose.pose.position.y = self._original_route_lane_data_xy[i,1]
            pose.pose.position.z = 40
            pose.pose.orientation.w = 1.0
            self._path0.poses.append(pose)

            pose = PoseStamped()
            pose.pose.position.x = self._shift_route_lane_data_xy[i, 0]
            pose.pose.position.y = self._shift_route_lane_data_xy[i, 1]
            pose.pose.position.z = 40
            pose.pose.orientation.w = 1.0
            self._path1.poses.append(pose)

        for topic, msg, t in bag.read_messages(topics = [agent_topic, ego_pose_topic, img_color_topic, markers_topic]):
            if topic == agent_topic:
                agent_time = t.to_sec()
                agent_flag = True
                agent_msg = msg
            if topic == img_color_topic and agent_flag:
                img_time = t.to_sec()
                dt = img_time - agent_time
                if dt > 0.10: continue
                img_flag = True
                img_msg = msg
            if topic == ego_pose_topic and agent_flag:
                ego_pose_time = t.to_sec()
                dt = ego_pose_time - agent_time
                # 时间同步
                if dt > 0.10: continue
                ego_pose_flag = True
                ego_pose_msg = msg

            if topic == markers_topic:
                marker_msg = msg

            if ego_pose_flag and agent_flag and img_flag:  #and (abs(last_ego_time - ego_pose_time)>0.005)
                ego_pose_flag, agent_flag, img_flag = False, False, False
                ego_poses_list.append(ego_pose_msg)
                ego_time_list.append(agent_time)
                agents_list.append(agent_msg)
                img_color_list.append(img_msg)
                markers_list.append(marker_msg)
                stamp_list.append(t)
        print('同步完成')
        print(len(ego_poses_list))

        for count_i in range(len(ego_poses_list)-self._num_future_poses-1):
            cur_header = Header()
            cur_header.stamp = stamp_list[count_i]
            cur_header.frame_id = "map"
            # print(ego_pose_flag,agent_flag, img_flag)

            self._ego_past_queue.append(ego_poses_list[count_i])
            self._agents_past_queue.append(agents_list[count_i])
            if (len(self._ego_past_queue) > 21)&(len(self._agents_past_queue) > 21):
                print('planing...')
                ego_state = StateSE2(x=self._ego_past_queue[-1].position.x, y=self._ego_past_queue[-1].position.y,
                                     heading=self._ego_past_queue[-1].euler_angles.z)
                ego_agent_past = get_ego_past_to_tensor_list(self._ego_past_queue)
                past_tracked_objects_tensor_list, past_tracked_objects_types = get_tracked_objects_to_tensor_list(
                    self._agents_past_queue)
                time_stamps_past = get_past_timestamps_to_tensor(self._ego_past_queue)
                ego_agent_past, neighbor_agents_past = agent_past_process(
                    ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list,
                    past_tracked_objects_types,
                    self._num_agents
                )

                self._future_path = Path()
                for j in range(self._num_future_poses):
                    pose = PoseStamped()
                    pose.pose.position.x = ego_poses_list[count_i + j + 1].position.x
                    pose.pose.position.y = ego_poses_list[count_i + j + 1].position.y
                    pose.pose.position.z = ego_poses_list[count_i + j + 1].position.z
                    pose.pose.orientation.x = ego_poses_list[count_i + j + 1].orientation.qx
                    pose.pose.orientation.y = ego_poses_list[count_i + j + 1].orientation.qy
                    pose.pose.orientation.z = ego_poses_list[count_i + j + 1].orientation.qz
                    pose.pose.orientation.w = ego_poses_list[count_i + j + 1].orientation.qw
                    self._future_path.poses.append(pose)

                distances = cdist(self._original_route_lane_data_xy,
                                  np.array([self._ego_past_queue[-1].position.x, self._ego_past_queue[-1].position.y]).reshape(1,-1))
                cur_ego_index = np.argmin(distances)  # 找到距离最小的点的索引
                min_index = int(cur_ego_index - self._radius / self._sample_interval + 1)
                max_index = int(cur_ego_index + self._radius / self._sample_interval)
                if (min_index < 0) | (max_index > self._original_route_lane_data_xy.shape[0] - 1): continue
                lanes_mid: List[List[Point2D]] = []
                baseline_path_polyline = [Point2D(self._original_route_lane_data_xy[min_index + node, 0],
                                                  self._original_route_lane_data_xy[min_index + node, 1]) for node in
                                          range(int(self._radius / self._sample_interval * 2))]
                lanes_mid.append(baseline_path_polyline)
                baseline_path_polyline = [Point2D(self._shift_route_lane_data_xy[min_index + node, 0],
                                                  self._shift_route_lane_data_xy[min_index + node, 1]) for node in
                                          range(int(self._radius / self._sample_interval * 2))]
                lanes_mid.append(baseline_path_polyline)
                coords_map_lanes_polylines = MapObjectPolylines(lanes_mid)
                coords_route_lanes_polylines = MapObjectPolylines(lanes_mid)
                crosswalk: List[List[Point2D]] = []
                coords_crosswalk_polylines = MapObjectPolylines(crosswalk)
                coords: Dict[str, MapObjectPolylines] = {}
                # extract generic map objects
                coords[self._map_features[0]] = coords_map_lanes_polylines
                coords[self._map_features[1]] = coords_route_lanes_polylines
                coords[self._map_features[2]] = coords_crosswalk_polylines
                traffic_light_encoding = np.zeros([2, 4], dtype=int)
                traffic_light_encoding[:, -1] = 1
                traffic_light_data_at_t: Dict[str, LaneSegmentTrafficLightData] = {}
                traffic_light_data: List[Dict[str, LaneSegmentTrafficLightData]] = []
                traffic_light_data_at_t[self._map_features[0]] = LaneSegmentTrafficLightData(
                    list(map(tuple, traffic_light_encoding)))
                traffic_light_data = traffic_light_data_at_t
                vector_map = map_process(ego_state, coords, traffic_light_data, self._map_features,
                                         self._max_elements, self._max_points, self._interpolation_method)

                data = {"ego_agent_past": ego_agent_past[1:],
                        "neighbor_agents_past": neighbor_agents_past[:, 1:]}
                data.update(vector_map)
                data = convert_to_model_inputs(data, self._device)

                with torch.no_grad():
                    # 这里和后续修改模型的代码有区别，修改模型后，这里的输出是6个，现在是5个
                    plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = self._planner._get_prediction(data)

                smooth_plan = plan[0].cpu().numpy()
                origin_pose = [ego_state.x, ego_state.y, ego_state.heading]
                absolute_poses = np.array(self.relative_to_absolute_poses(origin_pose, smooth_plan))

                path = Path(header=cur_header)
                for plani in range(absolute_poses.shape[0]):
                    posestamp = PoseStamped(header=cur_header)
                    pose = Pose()
                    pose.position.x = absolute_poses[plani, 0]
                    pose.position.y = absolute_poses[plani, 1]
                    pose.position.z = self._ego_past_queue[-1].position.z
                    q = quaternion_from_euler(0, 0, absolute_poses[plani, 2])
                    pose.orientation.x = q[0]
                    pose.orientation.y = q[1]
                    pose.orientation.z = q[2]
                    pose.orientation.w = q[3]
                    posestamp.pose = pose
                    path.poses.append(posestamp)
                self._planner_path_pub.publish(path)
                print('plan pub')

            pose = Pose()
            pose.position.x = self._ego_past_queue[-1].position.x
            pose.position.y = self._ego_past_queue[-1].position.y
            pose.position.z = self._ego_past_queue[-1].position.z
            pose.orientation.x = self._ego_past_queue[-1].orientation.qx
            pose.orientation.y = self._ego_past_queue[-1].orientation.qy
            pose.orientation.z = self._ego_past_queue[-1].orientation.qz
            pose.orientation.w = self._ego_past_queue[-1].orientation.qw
            markers = MarkerArray()
            marker = Marker(header=cur_header)
            marker.ns = 'mesh'
            marker.id = 0
            marker.type = Marker.MESH_RESOURCE
            marker.action = marker.ADD
            # mesh_resource需要用url进行赋值
            marker.mesh_resource = "file:///home/user/workspace/hjj/ws_perception/src/detected_objects_visualizer/models/car.dae"
            marker.pose = pose
            marker.scale = Vector3(1.0, 1.0, 1.0)
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
            markers.markers.append(marker)

            markers_msg = markers_list[count_i]
            if markers_msg is not None:
                for marker_msg in markers_msg.markers:
                    marker_msg.header = cur_header
                self._obj_markers_pub.publish(markers_msg)
            img_msg = img_color_list[count_i]
            img_msg.header = cur_header

            self._path0.header = cur_header
            self._path1.header = cur_header
            self._future_path.header = cur_header

            self._ego_markers_pub.publish(markers)
            self._img_pub.publish(img_msg)
            self._centerline_path_pub_0.publish(self._path0)
            self._centerline_path_pub_1.publish(self._path1)
            self._future_gt_path_pub.publish(self._future_path)

        # rospy.spin()

def main(args):
    rospy.init_node('xiaoba_rosbag_test')
    node = gameformer_planner(args.bag_path, args.model_path, args.device)
    node.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run xiaoba rosbag test')
    parser.add_argument('--bag_path', type=str, help='path to bag',
                        default="/data/datasets/xiaoba/2024.1.11/2024-01-11-17-20-37_part2_with_det_2.bag")
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--device', type=str, default='cuda', help='device to run model on')
    args = parser.parse_args()

    main(args)