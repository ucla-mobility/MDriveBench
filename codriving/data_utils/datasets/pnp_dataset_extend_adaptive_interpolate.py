from typing import List, Dict, Any, Iterable
import os
import copy
import re
import io
import logging
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

# from codriving.data_utils.datasets.basedataset import BaseIODataset
from .basedataset import BaseIODataset
from common.heatmap import generate_relative_heatmap, render_self_car
from common.detection import generate_det_data

from skimage.measure import block_reduce

from codriving import CODRIVING_REGISTRY

from opencood.data_utils.datasets import build_dataset

from scipy.interpolate import CubicSpline
import random
import copy


# CarlaMVDetDataset_planner
@CODRIVING_REGISTRY.register
class CarlaMVDatasetEnd2End_EXTEND_adaptive_interpolate(BaseIODataset):
    """Carla multi-vehicle dataset with gt occupancy

    """

    route_frames : List
    """records file paths for the ego vehicle.

    .. code-block::

        structure:
            list[
                tuple(
                    str, # verify the path
                    int  # verify the start frame
                ), ...
            ]
        example:
            [
				(/dataset_example/weather-0/data/routes_town05_long_w0_02_15_02_37_28/ego_vehicle_0, 0),
				(/dataset_example/weather-0/data/routes_town05_long_w0_02_15_02_37_28/ego_vehicle_0, 5)
            ]
    """

    def __init__(
        self,
        root,
        towns,
        weathers,
        input_frame : int=5,
        output_points : int=20,
        skip_frames : int=1,
        det_range : Iterable[int]=(36, 12, 12, 12, 0.125),
        perception_hypes : dict={},
        dataset_mode: str='train',
        dataset_index_list: str='dataset_index.txt',
        filte_danger: bool=False,
        skip_frame: bool=False,
    ):
        """
        Initialize Carla dataset. [KEY] contribution: self.route_frames.

        Args:
            towns: TODO (weibo)
            weathers: TODO (weibo)
            input_frame: TODO (weibo)
            output_points: TODO (weibo)
            det_range: TODO (weibo)
        """
        super().__init__()
        
        self.input_frame = input_frame # number of input frames
        self.output_points = output_points # number of output points
        self.skip_frames = skip_frames  # 1 control the nummber of frames to skip
        self.det_range = det_range # [front, back, left, right, resolution]
        self.distance_to_map_center = (self.det_range[0]+self.det_range[1])/2-self.det_range[1]
        self.with_drivable_area = True

        print('towns: ',towns)

        ##########
        ## Load file paths and build the route structure.
        ##########
        self.route_frames = []
        index = 0

        self.pixel_per_meter = int(det_range[-1]*8)
        self.max_distance = max(det_range)
        self.root = root

        print('data_index_name:', dataset_index_list)
        
        # if os.path.exists(os.path.join("processed_data", dataset_index_list.replace("txt", "json"))):
        #     os.remove(os.path.join("processed_data", dataset_index_list.replace("txt", "json")))
                
        self.original_waypoints = {}
        filter_file = None
        if filte_danger:
            if os.path.exists(os.path.join(self.root,'danger_frames.json')):
                filter_file = self._load_json(os.path.join(root, 'danger_frames.json'))
        if not dataset_index_list.startswith("/GPFS/"):
            dataset_indexs = self._load_text(os.path.join(root, dataset_index_list)).split('\n')
            # if os.path.exists(os.path.join("processed_data", dataset_index_list.replace("txt", "json"))):
            #     print("Loading existing waypoints")
            #     self.original_waypoints = self._load_json(os.path.join("processed_data", dataset_index_list.replace("txt", "json")))
        else:
            dataset_indexs = self._load_text(dataset_index_list).split('\n')
            
                
        pattern = re.compile('weather-(\d+).*town(\d\d)')
        line_num = 0
        
        for line in dataset_indexs:
            if len(line.split()) != 3:
                continue
            path, frames, egos = line.split()
            route_path = os.path.join(root, path)
            frames = int(frames)
            self.total_frames = frames
            res = pattern.findall(path)
            if len(res) != 1:
                continue
            weather = int(res[0][0])
            town = int(res[0][1])            
            if weather not in weathers or town not in towns:
                continue
            line_num += 1
            print('Line:', line_num, '/', len(dataset_indexs))
            files = os.listdir(route_path)
            ego_files = [file for file in files if file.startswith('ego')]
            rsu_files = [file for file in files if file.startswith('rsu')]

            # recompute rsu change frames
            file_len_list = []
            if len(rsu_files) > 0:
                for rsu_file in ['rsu_1000', 'rsu_1001']:
                    if rsu_file in rsu_files:
                        rsu_frame_len = len(os.listdir(os.path.join(route_path,rsu_file,'measurements')))
                        file_len_list.append(rsu_frame_len)
            try:
                self.rsu_change_frame = max(file_len_list) + 1
            except:
                continue
            for j, file in enumerate(ego_files):
                ego_path = os.path.join(route_path, file)

                others_list = ego_files[:j]+ego_files[j+1:]
                others_path_list = []
                for others in others_list:
                    others_path_list.append(os.path.join(path, others))
                
                # import ipdb; ipdb.set_trace()
                # if not os.path.exists(os.path.join("processed_data", dataset_index_list.replace("txt", "json"))):
                valid_frame_list = []
                invalid_frame_list = []
                waypoint_dict = {}
                for i in range(0, frames-input_frame*self.skip_frames-output_points*self.skip_frames, 1):
                    cumulative_distance = 0
                    mea_i = self._load_json(os.path.join(ego_path, "measurements", "%04d.json" % (i+(self.input_frame-1)*self.skip_frames)))
                    # mea_i = self._load_json(os.path.join(ego_path, "measurements", "%04d.json" % (i+(self.input_frame-1)*self.skip_frames)))
                    
                    ego_x = mea_i["gps_x"]
                    ego_y = mea_i["gps_y"]
                    ego_theta = mea_i["theta"]
                    speed = mea_i["speed"]
                    jj = 0
                    R = np.array(
                        [
                            [np.cos(ego_theta), -np.sin(ego_theta)],
                            [np.sin(ego_theta), np.cos(ego_theta)],
                        ]
                    )   
                    surrounding_vehicles_dis = []
                    import math
                    def global_to_local(x0, y0, theta, x1, y1):
                    # Left-handed coordinate system: theta is clockwise to -y, forward is y, left is x
                        dx = x1 - x0
                        dy = y1 - y0
                        psi_rad = math.pi - theta
                        
                        x_local = math.cos(psi_rad) * dx - math.sin(psi_rad) * dy
                        y_local = math.sin(psi_rad) * dx + math.cos(psi_rad) * dy
                        return x_local, y_local
                    actors_data = self._load_json(os.path.join(ego_path, "actors_data", "%04d.json" % (i+(self.input_frame-1)*self.skip_frames)))
                    for actor, actor_data in actors_data.items():
                        actor_x = actor_data['loc'][0]
                        actor_y = actor_data['loc'][1]
                        if actor_x == ego_x and actor_y == ego_y:
                            continue
                        if actor_data['tpe'] != 0 and actor_data['tpe'] != 1 and actor_data['tpe'] != 3:
                            continue
                        local_x, local_y = global_to_local(ego_x, ego_y, ego_theta, actor_x, actor_y)
                        if local_x < -12 or local_x > 12 or local_y < -12 or local_y > 36:
                            continue
                        dis = np.sqrt((actor_x - ego_x)**2 + (actor_y - ego_y)**2)
                        surrounding_vehicles_dis.append(dis)
                    if len(surrounding_vehicles_dis) > 0:
                        min_dis = min(surrounding_vehicles_dis)
                    else:
                        min_dis = 50
                    vehicles_num = len(surrounding_vehicles_dis) + 1
                    al, ah = 1, 5
                    a_dis = al + (ah - al) * (min_dis - 5) / 45
                    try:
                        a_den = 1/(1/ah + (1/al - 1/ah) * (vehicles_num - 1) / 19)
                    except ZeroDivisionError:
                        a_den = 0
                    w = 0.5
                    a_max = w*a_dis + (1-w)*a_den
                    try:
                        command_waypoints = []
                        command_waypoints.append(np.array([0, 0]))
                        while cumulative_distance < speed * 0.2 * 20 + 0.5 * a_max * (0.2 * 20) ** 2 + 1 or len(command_waypoints) < 20:
                            mea_j = self._load_json(os.path.join(ego_path, "measurements", "%04d.json" % (i+self.input_frame*self.skip_frames+jj)))
                            command_waypoints.append(R.T.dot(np.array([mea_j["gps_y"]-ego_y, -1*(mea_j["gps_x"]-ego_x)])))
                            cumulative_distance += np.linalg.norm(command_waypoints[-1] - command_waypoints[-2])
                            jj += self.skip_frames
                    except FileNotFoundError:
                        # invalid_frame_list.append(i)
                        # continue
                        break
        
                    # 将非常小的点归零
                    for kk in range(len(command_waypoints)):
                        for mm in range(len(command_waypoints[kk])):
                            if abs(command_waypoints[kk][mm]) < 0.001:
                                command_waypoints[kk][mm] = 0
                                
                    unique_points = [command_waypoints[0]]
                    for kk in range(1, len(command_waypoints), 1):
                        if abs(command_waypoints[kk][1]) > abs(unique_points[-1][1]):
                            unique_points.append(command_waypoints[kk])

                    if len(unique_points) < 2:
                        # print(f"frame {i} cannot be extended")
                        # invalid_frame_list.append(i)
                        continue
                        
                    valid_frame_list.append(i)
                    waypoint_dict[i] = np.array(command_waypoints)
                    # waypoint_dict[i] = command_waypoints
                    # LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
                    # if LOCAL_RANK == 0:
                        # print(f"There are {len(invalid_frame_list)} invalid frames in {frames} frames.")
                    # if True:
                    #     print('Invalid frames:', invalid_frame_list)
                self.original_waypoints[ego_path] = waypoint_dict
            

                # for i in range(0, frames-input_frame*self.skip_frames-output_points*self.skip_frames - 1, 1):
                for i in valid_frame_list:
                    # import ipdb; ipdb.set_trace()
                    if skip_frame:
                        if i%2 == 0:
                            continue
                    if filter_file is not None:
                        danger_frame_flag = False
                        for route_id in filter_file:
                            if route_path.endswith(filter_file[route_id]['sub_path']):
                                for junction_range in filter_file[route_id]['selected_frames'][file]:
                                    if i > junction_range[0] and i < junction_range[1]:
                                        danger_frame_flag = True
                        if (not danger_frame_flag):
                            continue
                    scene_dict = {}
                    scene_dict['ego'] = ego_path
                    scene_dict['other_egos'] = others_path_list
                    scene_dict['num_car'] = len(ego_files)
                    scene_dict['rsu'] = []
                    # order of rsu
                    if i%self.rsu_change_frame != 0  and len(rsu_files)>0:
                        order = int(i/self.rsu_change_frame)+1 #  int(i/10)+1 
                        rsu_path = 'rsu_{}00{}'.format(order, ego_path[-1])
                        if True: # os.path.exists(os.path.join(route_path, rsu_path,'measurements','{}.json'.format(str(i).zfill(4)))):
                            scene_dict['rsu'].append(os.path.join(path, rsu_path))
                    
                    
                    sp_list = [0, 1, 2, 3, 4]
                    
                    
                    for ex in sp_list:
                        scene_dict['extend'] = ex
                        # if index > 7350 and index < 7360:
                        self.route_frames.append((copy.deepcopy(scene_dict), i))
                        # index += 1
        # if not dataset_index_list.startswith("/GPFS/") and not os.path.exists(os.path.join("processed_data", dataset_index_list.replace("txt", "json"))):            
        #     with open(os.path.join("processed_data", dataset_index_list.replace("txt", "json")), 'w') as f:
        #         json.dump(self.original_waypoints, f, indent=4)
                        
                
                            
        logging.info("Planning test sub route dir nums: %d" % len(self.route_frames))

        ## insert perception dataset
        print('Perception Dataset Building')
        self.dataset_mode = dataset_mode
        dataset_mode = 'val'
        if dataset_mode == 'train':
            opencood_dataset = build_dataset(perception_hypes, visualize=False, train=True)
        elif dataset_mode == 'val':
            opencood_dataset = build_dataset(perception_hypes,
                                            visualize=False,
                                            train=False)
        elif dataset_mode == 'test':
            opencood_dataset = build_dataset(perception_hypes,
                                            visualize=True,
                                            train=False)            
        self.perception_dataset = opencood_dataset



    def __len__(self):
        return len(self.route_frames)


    def get_one_record(self,
            route_dir : str,
            frame_id : int,
            extend : int, 
            waypoints: list
        ) -> Dict[str, Any]:
        """Read data of one record

        Args:
            scene_dict: index given by dataloader
            frame_id: start frame id
            extend: speed_idx

        Return:
            batch data, including model input and training target
        """
        output_record = {
            'occupancy_map': [],   # T=5, H=20, W=40
            'occupancy_ego': [],   # T=5, H=20, W=40
            'occupancy_local_command': [],
            'command_waypoints': 0, # T=10, 2,
            'target': [],
            'drivable_area': None,
        }
        assert extend in [0, 1, 2, 3, 4], f"extend should be in [0, 1, 2, 3, 4], but got {extend}"
        assert len(waypoints >= 20), f"waypoints length should be larger than 20, but got {len(waypoints)}"
        if self.with_drivable_area:
            bev_image = self._load_image(os.path.join(route_dir, "birdview", "%04d.jpg" % (frame_id+(self.input_frame-1)*self.skip_frames)))
            img_c = bev_image.crop([140, 20, 260, 260])
            img_r = np.array(img_c.resize((96, 192))) # 96, 192, 3
            # print('Image r: ', img_r.shape)
            driavble_area = np.where(img_r.sum(axis=2)>200, 1, 0) # 192, 96, only 0/1.
            # print('Drivable area: ', driavble_area.shape)
            output_record['drivable_area'] = driavble_area


        ##########
        ## load environment data and generate temporal input sequence.
        ##########        
        measurements_last = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+(self.input_frame-1)*self.skip_frames)))
        actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % (frame_id+(self.input_frame-1)*self.skip_frames)))
        output_record["speed"] = measurements_last["speed"]

        # command
        # speed: stop, slow, hold, fast  # NOTE: 3=hold, 4=fast, keep consistent with later usage
        # direction: left, right, straight, lane follow, lane change left, lane change right
        cmd_direction = torch.zeros(6)
        cmd_speed = torch.zeros(4)
        gt_command = measurements_last["command"]
        cmd_id = int(gt_command)-1
        if cmd_id >=0 and cmd_id <= 5:
            cmd_direction[int(gt_command)-1] = 1
        if measurements_last["should_brake"]:
            speed_idx = 0
        elif measurements_last["should_slow"]:
            speed_idx = 1
        elif measurements_last["target_speed"]-measurements_last["speed"]>1.5:
            speed_idx = 3
        else:
            speed_idx = 2
        output_record["cmd_direction"] = cmd_direction
        if extend==0:
            cmd_speed[speed_idx] = 1
        else:
            cmd_speed[extend-1] = 1
        output_record["cmd_speed"] = cmd_speed

        ego_theta = measurements_last["theta"]
        ego_x = measurements_last["gps_x"]
        ego_y = measurements_last["gps_y"]

        x_command = measurements_last["x_command"]
        y_command = measurements_last["y_command"]

        egp_pos_last = {
            "x": ego_y,
            "y": -ego_x,
            "theta": ego_theta
        }

        R = np.array(
            [
                [np.cos(ego_theta), -np.sin(ego_theta)],
                [np.sin(ego_theta), np.cos(ego_theta)],
            ]
        )

        local_command_point = np.array([y_command - ego_y, -1*(x_command - ego_x)])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = np.clip(local_command_point, a_min=[-12, -36], a_max=[12, 12])
        local_command_point[np.isnan(local_command_point)] = 0
        # clip the point out of detection range
        
        # NOTE: we should take the ground-truth future position as the supervision!
        # instead of future waypopints which might not be reached.
        command_waypoints = [np.array(point) for point in waypoints]
        if extend != 0:
            surrounding_vehicles_dis = []
            import math
            def global_to_local(x0, y0, theta, x1, y1):
                dx = x1 - x0
                dy = y1 - y0
                psi_rad = math.pi - theta
                
                x_local = math.cos(psi_rad) * dx - math.sin(psi_rad) * dy
                y_local = math.sin(psi_rad) * dx + math.cos(psi_rad) * dy
                return x_local, y_local
            for actor, actor_data in actors_data.items():
                actor_x = actor_data['loc'][0]
                actor_y = actor_data['loc'][1]
                if actor_x == ego_x and actor_y == ego_y:
                    continue
                if actor_data['tpe'] != 0 and actor_data['tpe'] != 1 and actor_data['tpe'] != 3:
                    continue
                local_x, local_y = global_to_local(ego_x, ego_y, ego_theta, actor_x, actor_y)
                if local_x < -12 or local_x > 12 or local_y < -12 or local_y > 36:
                    continue
                dis = np.sqrt((actor_x - ego_x)**2 + (actor_y - ego_y)**2)
                surrounding_vehicles_dis.append(dis)
            if len(surrounding_vehicles_dis) > 0:
                min_dis = min(surrounding_vehicles_dis)
            else:
                min_dis = 50
            vehicles_num = len(surrounding_vehicles_dis) + 1
                
            output_record["original_waypoints"] = copy.deepcopy(command_waypoints)
            def accelerate_waypoints(ori_waypoints, a=2, degree=3):

                waypoints = ori_waypoints
                t = 1/5
                # delta = output_record["speed"]*t + 0.5 * a * t**2
                v = output_record["speed"]
                # random.seed(frame_id)
                # delta = delta if delta > 0.1 else random.uniform(0.1, 0.2)

                new_waypoints = []
                current_x = 0
                current_y = 0
                prev_x = 0
                prev_y = 0
                next_x = 0
                next_y = 0
                while len(new_waypoints) < 20:
                    v_new = v + a * t
                    t_hold = 0
                    if v_new < 0:
                        v_new = 0
                    elif v_new > 8:
                        v_new = 8
                        t_hold = (v_new - v) / a
                    if a != 0:
                        delta = (v_new ** 2 - v ** 2) / (2 * a) + v_new * t_hold
                    else:
                        delta = v * t 
                    v = v_new
                    current_length = 0
                    dd = 0
                    while current_length < delta:
                        if len(waypoints) == 0:
                            remaining_length = delta - current_length
                            next_x = prev_x + remaining_length * (next_x - prev_x) / np.linalg.norm([next_x - prev_x, next_y - prev_y])
                            next_y = prev_y + remaining_length * (next_y - prev_y) / np.linalg.norm([next_x - prev_x, next_y - prev_y])
                            print(f"{frame_id}'s route is too short! {len(new_waypoints)} waypoints generated. Others are extended along the last direction.")
                        else: 
                            next_x, next_y = waypoints.pop(0)
                        dd = np.linalg.norm([next_x - current_x, next_y - current_y])
                        prev_x, prev_y = current_x, current_y
                        current_x, current_y = next_x, next_y
                        current_length += dd
                    current_length -= dd
                    alpha = (delta - current_length) / dd if dd > 0 else 0 
                    waypoint_x = prev_x + alpha * (next_x - prev_x)
                    waypoint_y = prev_y + alpha * (next_y - prev_y)
                    new_waypoints.append([waypoint_x, waypoint_y])
                    waypoints.insert(0, [next_x, next_y])
                    current_x, current_y = waypoint_x, waypoint_y
                    prev_x, prev_y = current_x, current_y

                return new_waypoints

            def get_acceleration(min_dis, vehicles_num, extend):
                if min_dis < 5:
                    min_dis = 5
                if min_dis > 50:
                    min_dis = 50            
                if vehicles_num > 20:
                    vehicles_num = 20
                    
                if extend == 1:
                    al, ah = -10, -10
                elif extend == 2:
                    al, ah = -8, -1
                elif extend == 3:
                    al, ah = 0, 0
                elif extend == 4:
                    al, ah = 1, 5
                a_dis = al + (ah - al) * (min_dis - 5) / 45
                try:
                    a_den = 1/(1/ah + (1/al - 1/ah) * (vehicles_num - 1) / 19)
                except ZeroDivisionError:
                    a_den = 0
                w = 0.5
                return w*a_dis + (1-w)*a_den
            
            acceleration = get_acceleration(min_dis, vehicles_num, extend)
            
            output_record["acceleration"] = acceleration
            command_waypoints = accelerate_waypoints(command_waypoints, acceleration)
        else:
            command_waypoints = command_waypoints[:20]
        assert len(command_waypoints) == self.output_points, f"command_waypoints length is {len(command_waypoints)}"
        output_record["command_waypoints"] = torch.from_numpy(np.array(command_waypoints))

        detmap_pose = []
        for frame_cur in range(0, self.input_frame*self.skip_frames, self.skip_frames):
            measurements = self._load_json(os.path.join(route_dir, "measurements", "%04d.json" % (frame_id+frame_cur)))
            detmap_pose_x = measurements['lidar_pose_x'] + self.distance_to_map_center*np.cos(measurements["theta"]-np.pi/2)
            detmap_pose_y = measurements['lidar_pose_y'] + self.distance_to_map_center*np.sin(measurements["theta"]-np.pi/2)
            detmap_theta = measurements["theta"] + np.pi/2
            detmap_pose.append(np.array([-detmap_pose_y, detmap_pose_x, detmap_theta]))

            actors_data = self._load_json(os.path.join(route_dir, "actors_data", "%04d.json" % (frame_id+frame_cur)))

            heatmap = generate_relative_heatmap(
                copy.deepcopy(measurements), copy.deepcopy(actors_data), egp_pos_last,
                pixels_per_meter=8,
                max_distance=self.max_distance
            )
            # max_distance(32) * pixel_per_meter(8) * 2, max_distance*pixel_per_meter*2
            occ_map = block_reduce(heatmap, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
            # occ_map: 75, 75
            occ_map = np.clip(occ_map, 0.0, 255.0)
            # TODO: change it into auto calculation
            occ_map = occ_map[:48*8, 48*4:48*8]
            # Verify the detection range.
            output_record["occupancy_map"].append(occ_map)

            # calculate the position of past frames according to current pos.
            self_car_map = render_self_car( 
                loc=np.array(R.T.dot(np.array(
                        [measurements["gps_y"]-ego_y, -1*(measurements["gps_x"]-ego_x)]
                        ))),
                ori=np.array([np.sin(measurements["theta"]-ego_theta), -np.cos(measurements["theta"]-ego_theta)]),
                box=np.array([2.45, 1.0]),
                color=[1, 1, 0], 
                pixels_per_meter=8,
                max_distance=self.max_distance
            )[:, :, 0]
            
            self_car_map = block_reduce(self_car_map, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
            self_car_map = np.clip(self_car_map, 0.0, 255.0)
            self_car_map = self_car_map[:48*8, 48*4:48*8]
            output_record["occupancy_ego"].append(self_car_map)	


            local_command_map = render_self_car( 
                loc=local_command_point,
                ori=np.array([0, 1]),
                box=np.array([1.0, 1.0]),
                color=[1, 1, 0], 
                pixels_per_meter=8,
                max_distance=self.max_distance
            )[:, :, 0]
            
            local_command_map = block_reduce(local_command_map, block_size=(self.pixel_per_meter, self.pixel_per_meter), func=np.mean)
            local_command_map = np.clip(local_command_map, 0.0, 255.0)
            local_command_map = local_command_map[:48*8, 48*4:48*8]
            output_record["occupancy_local_command"].append(local_command_map)	

        coordinate_map = np.ones((5, 2,)+local_command_map.shape)
        for h in range(local_command_map.shape[0]):
            coordinate_map[:, 0, h, :] *= h*self.det_range[-1]-self.det_range[0]
        for w in range(local_command_map.shape[1]):
            coordinate_map[:, 1, :, w] *= w*self.det_range[-1]-self.det_range[2]
        output_record["coordinate_map"] = coordinate_map
        output_record["target"] = local_command_point

        output_record['detmap_pose'] = detmap_pose

        return output_record




    def __getitem__(self, idx : int, data_dir=None) -> Dict:
        """
        Given the index, return the corresponding data. 

        Args:
            idx: index given by dataloader.

        Return:
            output_dict: batched data in the following format:

                .. code-block::

                    structure: list[
                        dict{
                        }, ... # see details for this dict in `get_one_record`
                    ] # len = 'num_cars'
        """
        
        if idx is not None:
            scene_dict, frame_id = self.route_frames[idx]
            scene = scene_dict['ego']
        elif data_dir is not None:
            scene, frame_id = os.path.join(self.root, data_dir[0]['ego']), data_dir[1]

        output_dict = self.get_one_record(scene, frame_id, extend=scene_dict['extend'], waypoints=self.original_waypoints[scene][frame_id])
        if output_dict is None:
            return None

        ## load perception model data
        import copy
        output_dict['perception_dict'] = {}
        for frame_cur in range(0, self.input_frame*self.skip_frames, self.skip_frames):
            scene_dict_to_load = copy.deepcopy(scene_dict)
            scene_dict_to_load['ego'] = scene_dict_to_load['ego'].replace(self.root, '')
            
            if 'rsu' in scene_dict.keys():
                if (frame_id+frame_cur)%self.rsu_change_frame == 0:
                    scene_dict_to_load['rsu'] = []
                else:
                    for i, rsu_path in enumerate(scene_dict_to_load['rsu']):
                        true_rsu_order = (frame_id+frame_cur)//self.rsu_change_frame + 1
                        rsu_order = int(rsu_path.split('/')[-1].split('_')[1][:-3])
                        if rsu_order != true_rsu_order:
                            last_len = len(rsu_path.split('_')[-1])
                            scene_dict_to_load['rsu'][i] = rsu_path[:-last_len] + str(true_rsu_order) + rsu_path.split('_')[-1][-3:]

            perception_dict = self.perception_dataset.__getitem__(idx=None, data_dir=(scene_dict_to_load, (frame_id+frame_cur)))
            output_dict['perception_dict'][frame_cur] = perception_dict
            output_dict['scene'] = scene

        return output_dict

    # @classmethod
    def collate_fn(self, batch : List, extra_source=None) -> tuple: # cls, 
        """Re-collate a batch.

        Args:
            batch: a batch of data len(batch)=batch_size

                - batch[i]: the ith data in the batch
                - batch[i][j]: the jth car in batch[i], batch[i][0] always center ego

                NOTE: BN = Σ_{i=0}^{B-1}(N_i)  # N_i is the num_car of the ith data in the batch

        Returns:
            batch data

                - model input: dict{'occupancy': torch.Tensor, size [B, 5, 2, 40, 20]},
                - model target: tuple(command_waypoints: torch.Tensor, size [B, 10])

        """
        # batch = [item for item in batch if item is not None and item["command_waypoints"] is not None]
        
        output_dict = {}
        occupancy = []
        future_waypoints = []
        target = []
        detmap_pose = []
        cmd_direction = []
        cmd_speed = []
        speed = []

        for i in range(len(batch)):
            occ_map = torch.from_numpy(np.array(batch[i]["occupancy_map"]))
            occ_ego = torch.from_numpy(np.array(batch[i]["occupancy_ego"]))
            occ_tar = torch.from_numpy(np.array(batch[i]["occupancy_local_command"]))
            occ_cor = torch.from_numpy(np.array(batch[i]["coordinate_map"]))
            occ_da  = torch.from_numpy(np.array(batch[i]["drivable_area"]))
            occ_img = torch.cat((occ_map.unsqueeze(1), occ_ego.unsqueeze(1), occ_tar.unsqueeze(1), occ_cor), dim=1)[:, :, ::2, ::2]
            if extra_source is not None:
                occ_img[:,0] = extra_source['occ'][0,:,0,:,:]
            occ_final = torch.cat((occ_img, occ_da.unsqueeze(0).unsqueeze(0).repeat(5, 1, 1, 1)), dim=1)
            occupancy.append(occ_final) # T=5, C=1+1+1+2+1, H, W
            future_waypoints.append(batch[i]["command_waypoints"])
            target.append(torch.from_numpy(np.array(batch[i]["target"])))
            detmap_pose.append(torch.from_numpy(np.array(batch[i]['detmap_pose'])))
            cmd_direction.append(batch[i]["cmd_direction"])
            cmd_speed.append(batch[i]["cmd_speed"])
            speed.append(batch[i]["speed"])

        # print(len(target), target[0].shape)
        output_dict["occupancy"] = torch.stack(occupancy, dim=0).float()
        output_dict["target"] = torch.stack(target, dim=0).float()
        future_waypoints = torch.stack(future_waypoints, dim=0).float()
        output_dict["future_waypoints"] = future_waypoints
        output_dict["detmap_pose"] = torch.stack(detmap_pose, dim=0).float()

        output_dict["cmd_direction"] = torch.stack(cmd_direction, dim=0).float()
        output_dict["cmd_speed"] = torch.stack(cmd_speed, dim=0).float()
        output_dict["speed"] = torch.tensor(speed).float()
        output_dict["scenes"] = [item['scene'] for item in batch]

        ## perception data
        perception_batch_dict = {}
        # for frame in frames:
        for frame in range(self.input_frame):
            batch_perception_list = []
            for i in range(len(batch)):
                batch_perception_list.append(batch[i]['perception_dict'][frame])
            if self.dataset_mode=='test':
                perception_batch_dict[frame] = self.perception_dataset.collate_batch_test(batch_perception_list, online_eval_only=True)
            else:
                perception_batch_dict[frame] = self.perception_dataset.collate_batch_train(batch_perception_list, online_eval_only=True)


        return output_dict, perception_batch_dict



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config-file",
        default="",
        type=str,
        metavar="FILE",
        help="Config file for training",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="#workers for dataloader (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval",
    )
    parser.add_argument(
        "--out-dir",
        default="./output/",
        type=str,
        help="directory to output model/log/etc.",
        )
    parser.add_argument(
        "--log-filename",
        default="log",
        type=str,
        help="log filename",
        )
    parser.add_argument(
        "--resume",
        default='',
        type=str,
        help="Path of the checkpoint from which the training resumes",
        )

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", type=str, default=False,
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parse_args = parser.parse_args()

    return parse_args

# python codriving/data_utils/datasets/pnp_dataset_extend.py --config-file ./codriving/hypes_yaml/codriving/end2end_covlm_cmd_extend.yaml --out-dir /GPFS/data/changxingliu/V2Xverse/ckpt --resume None --model_dir /GPFS/data/gjliu-1/Auto-driving/OpenCOODv2/opencood/logs/closed_loop/where2comm --log-filename log
def main():
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    args = parse_args()
    perception_hypes = yaml_utils.load_yaml(None, args)
    dataset = CarlaMVDatasetEnd2End_EXTEND_adaptive_interpolate(
        root='external_paths/data_root',
        towns=[5],
        weathers=[0],
        input_frame=5,
        output_points=10,
        skip_frames=1,
        det_range=(36, 12, 12, 12, 0.125),
        perception_hypes=perception_hypes,
        dataset_mode='train',
        dataset_index_list='dataset_index.txt',
        filte_danger=False,
        skip_frame=False,
    )

    import matplotlib.pyplot as plt
    import random

    color_list = ['r', 'g', 'b', 'y', 'c']
    random_indices = random.sample(range(10001), 100)

    for idx, k in enumerate(random_indices):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4)) 
        axes = axes.flatten() 

        for i, ax in enumerate(axes):
            ax.set_xlim(-5, 5)
            ax.set_ylim(-30, 0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            data = dataset.get_one_record(dataset.route_frames[k][0]['ego'], dataset.route_frames[k][1], i, dataset.original_waypoints[dataset.route_frames[k][0]['ego']][dataset.route_frames[k][1]])
            waypoints = data["command_waypoints"]
            speed_idx = np.where(data['cmd_speed'] == 1)[0]

            print(f"Processing route frame {k}, sub-plot {i+1}")

            ax.plot(waypoints[:, 0], waypoints[:, 1], 'o-', color=color_list[i % len(color_list)])
            ax.set_title(f'Waypoints {i+1} - {speed_idx}')

        plt.tight_layout()
        os.makedirs('/root/zijunwang/V2Xverse/ckpt/test', exist_ok=True)
        plt.savefig(f'/root/zijunwang/V2Xverse/ckpt/test/waypoints_{k}.jpg')
        plt.close() 


if __name__ == "__main__":
    main()


