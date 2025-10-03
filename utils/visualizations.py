import os
from habitat.utils.visualizations import maps
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
IMAGE_DIR = os.path.join("examples", "images")
MAP_THICKNESS_SCALAR: int = 128

def get_topdown_map(sim):
    top_down_map = maps.get_topdown_map_from_sim(
        sim, map_resolution=1024
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    return top_down_map

def draw_path(top_down_map , path_point):
    return maps.draw_path(top_down_map , path_point , color = maps.MAP_SHORTEST_PATH_COLOR)

def draw_points_on_map(topdown_map, points=None):
    img = topdown_map.copy()
    
    if points is not None and len(points) > 0:
        for pt in points:
            cv2.circle(img, tuple(np.array(pt, dtype=np.int32)), radius=5, color=(255, 0, 0), thickness=-1)  # 红色填充点

    return img

def draw_sound(sim , top_down_map ,point_type):
    position = sim.get_sound_state()
    t_x, t_y = maps.to_grid(
        position[2],
        position[0],
        (top_down_map.shape[0], top_down_map.shape[1]),
        sim=sim,
    )
    point_padding = 2 * int(
            np.ceil(512 / MAP_THICKNESS_SCALAR)
    )
    top_down_map[
        t_x - point_padding : t_x + point_padding + 1,
        t_y - point_padding : t_y + point_padding + 1,
    ] = point_type

def draw_agent(
    image: np.ndarray,
    agent_center_coord: Tuple[int, int],
    agent_rotation: float,
    agent_radius_px: int = 5,
    ):
    return maps.draw_agent(image ,agent_center_coord , agent_rotation , agent_radius_px)

def real_point_to_grid(sim , path_points , top_down_map):
    grid_point = []
    for point in path_points:
        grid_point.append(maps.to_grid(point[2] , point[0]  , (top_down_map.shape[0] , top_down_map.shape[1]) , sim  , sim.pathfinder))
    return grid_point

def draw_map(env, path_points):
        import math
        from scipy.spatial.transform import Rotation as R
        sim = env._env._sim
        agent_rot = sim.get_agent_state().rotation
        top_down_map = get_topdown_map(sim)
        path_points = real_point_to_grid(sim , path_points , top_down_map)
        quat = [agent_rot.x, agent_rot.y, agent_rot.z, agent_rot.w]
        r = R.from_quat(quat)
        euler = r.as_euler('zyx', degrees=True)
        yaw = euler[0]   # z 轴旋转角度
        
        # 保证在 0~360 之间
        yaw = (yaw + 360) % 360
        draw_sound(sim,top_down_map , maps.MAP_VIEW_POINT_INDICATOR)
        maps.draw_path(top_down_map, path_points)
        top_down_map = maps.draw_agent(
            top_down_map, path_points[-1], yaw, agent_radius_px=8
        )
        return top_down_map