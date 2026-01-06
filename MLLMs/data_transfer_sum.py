import json
import numpy as np
import os
import math
import json
from collections import Counter


def command_pair(command_num):
    if command_num == 1:
        return 'turn left at intersection'
    elif command_num == 2:
        return 'turn right at intersection'
    elif command_num == 3:
        return 'go straight at intersection'
    elif command_num == 4:
        return 'follow the lane'
    elif command_num == 5:
        return 'left lane change'
    elif command_num == 6:
        return 'right lane change'
    else:
        return ''

def speed_pair(speed, target_speed, should_slow, should_brake):
    if should_brake:
        return 'stop'
    if should_slow:
        return 'speed down'
    if speed > target_speed + 0.5:
        return 'speed down'
    elif speed < target_speed - 0.5:
        return 'speed up'
    else:
        return 'speed hold'


with open('/GPFS/public/dataset_cop3_lidarmini/dataset_index.txt', 'r') as file:
    data = file.read()
    lines = data.split('\n')
    extracted_data = []
    for line in lines:
        if line:
            directory, scene_count, agent_count = line.split(' ')
            extracted_data.append((directory, int(scene_count), int(agent_count)))

sumfile_path = "/GPFS/data/changxingliu/InternVL/data/data_sum_1030_intention_w_perception.json"
future_skip_frames = 1
output_points = 10
data_sum = []
for directory, scene_count, agent_count in extracted_data:
    directory = os.path.join('/GPFS/public/dataset_cop3_lidarmini/', directory)
    ego_veh_list = []
    for i in range(2):
        if os.path.exists(os.path.join(directory, f'ego_vehicle_{i}')):
            ego_veh_list.append(f'ego_vehicle_{i}')
    for ego_veh in ego_veh_list:
        measurement_path = os.path.join(directory, ego_veh, 'measurements/')
        camera_path = os.path.join(directory, ego_veh, 'rgb_front/')
        actors_path = os.path.join(directory, ego_veh, 'actors_data/')

        for frame in range(scene_count-output_points*future_skip_frames):
            measurement_file = os.path.join(measurement_path, f'{frame:04d}.json')
            camera_file = os.path.join(camera_path, f'{frame:04d}.jpg')
            actors_file = os.path.join(actors_path, f'{frame:04d}.json')
            with open(measurement_file, 'r') as file:
                measurement = json.load(file)
            with open(actors_file, 'r') as file:
                actors_data = json.load(file)

            data_sample = {"images": [camera_file]}

            danger_flag = 1
            if (measurement['command'] == 4 and not(measurement['should_brake'] or measurement['should_slow'])):
                danger_flag = 0

            speed = measurement['speed']
            navigation_ins = command_pair(measurement['command'])
            speed_ins = speed_pair(measurement['speed'], measurement['target_speed'], measurement['should_slow'], measurement['should_brake'])

            ego_theta = measurement["theta"]
            ego_x = measurement["gps_x"]
            ego_y = measurement["gps_y"]
            R = np.array(
                [
                    [np.cos(ego_theta), -np.sin(ego_theta)],
                    [np.sin(ego_theta), np.cos(ego_theta)],
                ]
            )
            x_command = measurement["x_command"]
            y_command = measurement["y_command"]
            local_command_point = np.array([y_command - ego_y, -1*(x_command - ego_x)])
            local_command_point = R.T.dot(local_command_point)
            local_command_point = np.clip(local_command_point, a_min=[-12, -36], a_max=[12, 12])
            local_command_point[np.isnan(local_command_point)] = 0
            target_point = local_command_point.tolist()

            # command_waypoints = []
            # for i in range(0, output_points*future_skip_frames, future_skip_frames):
            #     measurements_i = json.load(open(os.path.join(measurement_path, "%04d.json" % (frame + future_skip_frames + i))))
            #     command_waypoints.append(R.T.dot(np.array([measurements_i["gps_y"]-ego_y, -1*(measurements_i["gps_x"]-ego_x)])).tolist())

            surrounding_vehicles = []
            for actor, actor_data in actors_data.items():
                actor_x = actor_data['loc'][0]
                actor_y = actor_data['loc'][1]
                if actor_x == measurement['x'] and actor_y == measurement['y']:
                    continue
                if actor_data['tpe'] != 0 and actor_data['tpe'] != 1 and actor_data['tpe'] != 3:
                    continue
                if np.sqrt((actor_x - measurement["x"])**2 + (actor_y - measurement["y"])**2) > 50:
                    continue
                actor_speed = round(np.sqrt(actor_data['vel'][0]**2 + actor_data['vel'][1]**2), 2)
                actor_theta = math.atan2(actor_data['ori'][0], actor_data['ori'][1]) + math.pi
                surrounding_vehicles.append({'actor_id': actor, 'x': round(actor_x, 2), 'y': round(actor_y, 2), 'theta': actor_theta, 'speed': actor_speed, 'type': actor_data['tpe']})
                
            data_sample['direction intention'] = navigation_ins
            data_sample['speed intention'] = speed_ins
            data_sample['speed'] = round(speed, 2)
            data_sample['target speed'] = round(measurement['target_speed'], 2)
            data_sample['target point'] = [round(target_point[0], 2), round(target_point[1], 2)]
            data_sample['self_x'] = round(measurement["x"], 2)
            data_sample['self_y'] = round(measurement["y"], 2)
            data_sample['self_theta'] = measurement["theta"]
            data_sample['surrounding vehicles'] = surrounding_vehicles
            # data_sample['future waypoints'] = [[round(point[0], 2), round(point[1], 2)] for point in command_waypoints]

            if danger_flag:
                data_sample['danger'] = 1
            else:
                data_sample['danger'] = 0

            data_sum.append(data_sample)
        print('Done ', directory)
with open(sumfile_path, "w") as file:
    json.dump(data_sum, file, indent=4)

print('total length:', len(data_sum))


with open(sumfile_path, "r") as file:
    data_sum=json.load(file)
scene_data = {}
for record in data_sum:
    image_path = record['images'][0]
    scene = image_path.split('/')[-4]+'/'+image_path.split('/')[-3]
    frame = image_path.split('/')[-1].split('.')[0]
    
    if scene not in scene_data:
        scene_data[scene] = {'frames': [], 'speed': [], 'image_path': []}
    scene_data[scene]['frames'].append(int(frame))
    scene_data[scene]['speed'].append(record['speed'])
    scene_data[scene]['image_path'].append(image_path)

for scene, data in scene_data.items():
    refined_data = []
    frames = data['frames']
    speeds = data['speed']
    image_paths = data['image_path']
    
    # smooth the speed data
    window_length = 5
    smoothed_speeds = np.convolve(speeds, np.ones(window_length)/window_length, mode='same')

    # get intention
    intention_from_speed = []
    for s in smoothed_speeds:
        if s <= 1:
            intention_from_speed.append(1)
        elif 1 < s <= 5:
            intention_from_speed.append(2)
        else:
            intention_from_speed.append(3)

    filtered_intention = []
    current_group = []
    for i, val in enumerate(intention_from_speed):
        if not current_group or current_group[-1] == val:
            current_group.append(val)
        else:
            if len(current_group) <= 5:
                filtered_intention.extend([filtered_intention[-1]] * len(current_group))
            else:
                filtered_intention.extend(current_group)
            current_group = [val]
    if len(current_group) < 5:
        filtered_intention.extend([filtered_intention[-1]] * len(current_group))
    else:
        filtered_intention.extend(current_group)

    adjust_intention = []
    for i in range(len(filtered_intention)):
        if i+5>len(filtered_intention)-1:
            adjust_intention.append(filtered_intention[-1])
        else:
            adjust_intention.append(filtered_intention[i+5])

    level5_intention = []
    for i, (ori_i, adj_i) in enumerate(zip(filtered_intention, adjust_intention)):
        if ori_i == 1 and adj_i != 1:
            level5_intention.append(4) # start
        elif ori_i != 1 and adj_i == 1:
            level5_intention.append(5) # brake
        else:
            level5_intention.append(adj_i)
    # Count the occurrences of each number in level5_intention
    occurrences = Counter(level5_intention)
    print(occurrences[1], occurrences[2], occurrences[3], occurrences[4], occurrences[5])
        
    for frame, image_path in zip(frames, image_paths):
        if frame>=len(frames)-6: continue
        refined_data.append({
            'smoothed_speed': smoothed_speeds[frame],
            'smoothed_speed_1s': smoothed_speeds[frame+5],
            'intention_from_speed': intention_from_speed[frame],
            'filtered_intention': filtered_intention[frame],
            'intention_3': adjust_intention[frame],
            'intention_5': level5_intention[frame],
            'image_path': image_path
        })

    for record in data_sum:
        for refined_record in refined_data:
            if record['images'][0] == refined_record['image_path']:
                record['refined_intention_3'] = refined_record['intention_3']
                record['refined_intention_5'] = refined_record['intention_5']
                record['smoothed_speed'] = refined_record['smoothed_speed']
                record['smoothed_speed_1s'] = refined_record['smoothed_speed_1s']
                record['intention_from_speed'] = refined_record['intention_from_speed']
                record['filtered_intention'] = refined_record['filtered_intention']
                import ipdb; ipdb.set_trace()
                break

with open(sumfile_path, 'w') as f:
    json.dump(data_sum, f, indent=4)

print("All processed.")