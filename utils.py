from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from matplotlib import colors
from mcap_protobuf.writer import Writer
from google.protobuf.message import Message
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Color_pb2 import Color


@dataclass
class TopicWithData:
    topic_name: str
    timestamps: np.ndarray
    messages: List[Message]


def write_mcap_file(filename: str, topics: List[TopicWithData]):

    current_unixtime = 1690970130

    with open(filename, "wb") as f:
        with Writer(f) as mcap_writer:

            for topic in topics:

                num_samples = len(topic.timestamps)

                for k in range(num_samples):
                    timestamp = topic.timestamps[k]
                    timestamp_ns = int(timestamp*1e9) + \
                        int(current_unixtime*1e9)
                    mcap_writer.write_message(topic=topic.topic_name,
                                              message=topic.messages[k],
                                              log_time=timestamp_ns,
                                              publish_time=timestamp_ns)


def create_cut_in_trajectories() -> Tuple[pd.DataFrame, pd.DataFrame]:

    # general settings
    T = 10
    dt = 0.1
    timesteps = np.arange(0, T, dt)
    lane_width_meters = 4.0

    # --- Vehicle 1 ---
    v1_velocity = 30
    v1_pos_x = v1_velocity*timesteps
    v1_pos_y = np.zeros_like(v1_pos_x)

    pos_arr = np.c_[v1_pos_x, v1_pos_y]
    delta_pos_arr = np.diff(pos_arr, axis=0)
    v1_heading = np.arctan2(delta_pos_arr[:, 1], delta_pos_arr[:, 0])
    v1_heading = np.r_[v1_heading, v1_heading[-1]]

    df_veh_1 = pd.DataFrame({"x": v1_pos_x, "y": v1_pos_y,
                             "theta": v1_heading},
                            index=timesteps)

    # --- Vehicle 2 ---
    time_cut_in = 5.0
    v2_velocity = 35
    v2_pos_x = v2_velocity*timesteps
    v2_pos_y = lane_width_meters - \
        lane_width_meters*sigmoid(2*(timesteps-time_cut_in))

    pos_arr = np.c_[v2_pos_x, v2_pos_y]
    delta_pos_arr = np.diff(pos_arr, axis=0)
    v2_heading = np.arctan2(delta_pos_arr[:, 1], delta_pos_arr[:, 0])
    v2_heading = np.r_[v2_heading, v2_heading[-1]]

    df_veh_2 = pd.DataFrame({"x": v2_pos_x, "y": v2_pos_y,
                             "theta": v2_heading},
                            index=timesteps)

    return df_veh_1, df_veh_2


def create_vehicle_cuboid(x: float, y: float, z: float, theta: float,
                          color: str, alpha: float = 1.0) -> CubePrimitive:

    # vehicle size
    VEHICLE_LENGTH = 3.4
    VEHICLE_WIDTH = 1.8
    VEHICLE_HEIGHT = 1.5

    rot = Rotation.from_euler("zyx", [theta, 0, 0], degrees=False)
    quat = rot.as_quat()
    quat_foxglove = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    position_foxglove = Vector3(x=x, y=y, z=z + VEHICLE_HEIGHT/2)
    pose_foxglove = Pose(position=position_foxglove, orientation=quat_foxglove)

    size_foxglove = Vector3(
        x=VEHICLE_LENGTH, y=VEHICLE_WIDTH, z=VEHICLE_HEIGHT)

    r, g, b = colors.to_rgb(color)
    color_foxglove = Color(r=r, g=g, b=b, a=alpha)

    cuboid_foxglove = CubePrimitive(pose=pose_foxglove, size=size_foxglove,
                                    color=color_foxglove)
    return cuboid_foxglove


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
