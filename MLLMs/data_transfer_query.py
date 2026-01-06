import json
import numpy as np
import os
import json
import math


def judge_direction(self, x0, y0, theta0, x1, y1, theta1):
    # 计算两车之间的相对位置
    theta0 = self.normalize_angle(-theta0)
    theta1 = self.normalize_angle(-theta1)
    dx = x1 - x0
    dy = y0 - y1
    # 转换到对车的局部坐标系
    dx_local = dx * math.cos(theta1-math.pi) + dy * math.sin(theta1-math.pi)
    dy_local = -dx * math.sin(theta1-math.pi) + dy * math.cos(theta1-math.pi)
    
    # 计算本车相对于对车的角度
    relative_angle = self.normalize_angle(math.atan2(dx_local, dy_local))
    # 根据角度差异判断方位
    if -math.pi / 8 <= relative_angle <= math.pi / 8:
        direction = 'front'
    elif math.pi / 8 < relative_angle <= 3 * math.pi / 8:
        direction = 'right front'
    elif 3 * math.pi / 8 < relative_angle <= 5 * math.pi / 8:
        direction = 'right'
    elif 5 * math.pi / 8 < relative_angle <= 7 * math.pi / 8:
        direction = 'right rear'
    elif 7 * math.pi / 8 < relative_angle <= math.pi:
        direction = 'rear'
    elif -math.pi <= relative_angle < -7 * math.pi / 8:
        direction = 'rear'
    elif -7 * math.pi / 8 <= relative_angle < -5 * math.pi / 8:
        direction = 'left rear'
    elif -5 * math.pi / 8 <= relative_angle < -3 * math.pi / 8:
        direction = 'left'
    elif -3 * math.pi / 8 <= relative_angle < - math.pi / 8:
        direction = 'left front'

    # 计算heading
    angle_diff = self.normalize_angle(theta1 - theta0)
    if -math.pi / 4 <= angle_diff <= math.pi / 4:
        heading = 'forward'
    elif math.pi / 4 < angle_diff <= 3 * math.pi / 4:
        heading = 'right'
    elif -3 * math.pi / 4 <= angle_diff < -math.pi / 4:
        heading = 'left'
    else:
        heading = 'backward'

    return direction, heading

def global_to_local(x0, y0, theta, x1, y1):
    # 左手坐标，theta顺时针到-y，前y左x
    dx = x1 - x0
    dy = y1 - y0
    psi_rad = math.pi - theta
    
    x_local = math.cos(psi_rad) * dx - math.sin(psi_rad) * dy
    y_local = math.sin(psi_rad) * dx + math.cos(psi_rad) * dy
    return x_local, y_local

system = '''Suppose you are an autopilot assistant driving a car on the road. You will receive images from the car's front camera and are expected to provide driving intentions. There are other traffic participants in the scenario, and you may have communication with them. Your analysis logic chain should be as follows:
1. Understand the direction of the road and your own position.
2. Perceive surrounding objects.
3. Pay attention to key objects and dangerous situations.
4. Follow the rules listed below.
5. Check communication decision.
6. Finally, conclude the situation and provide driving intentions.

### Rules
1. If the environment is safe and clear, drive fast
2. Maintain a safe distance from the car in front.
3. Stop to avoid pedestrians preparing to cross the road.
4. Slow down or stop when other vehicles change lanes, merge or turn.
5. Slow down or stop when there is obstacle on the road ahead.
6. When establishing communication with other vehicles, take the communication decision as important reference.

### Real-time Inputs
Target direction: -nvins-
Current Speed: -speed- m/s

### Output Requirements
Provide the speed intention, include STOP, FASTER, SLOWER, KEEP.'''
# [start up; brake; stay; normal speed; high speed]
# Car/people position are provided in format [x, y, speed; ...]. +x is left, +y is forward, speed unit is m/s.

sumfile_path = "/GPFS/data/changxingliu/InternVL/data/data_sum_1030_intention_w_perception.json"

with open(sumfile_path, "r") as file:
    data_sum = json.load(file)
import random
random.shuffle(data_sum)

data_file = '/GPFS/data/changxingliu/InternVL/data/data_50k_intention_speed_4_wo_perception_short_answer_refined.json'
dataset = data_sum#[:10000]

data = []
intention_count = [0,0,0,0]  # [0, 0, 0, 0, 0]
# count_max = [15000, 28000, 17000] # [10000, 20000, 15000, 4680, 4520]
for k, scene in enumerate(dataset):
    # perception
    if math.isnan(scene['self_theta']): continue
    perc = [[], [], []]
    for actor in scene['surrounding vehicles']:
        local_x, local_y = global_to_local(scene['self_x'], scene['self_y'], scene['self_theta'], actor['x'], actor['y'])
        if local_x < -12 or local_x > 12 or local_y < -12 or local_y > 36:
            continue
        perc_i = 0 if actor['type']==0 else 1 if actor['type']==1 else 2
        perc[perc_i].append([local_x, local_y])
    
    # perception_str = 'Perceived Car: [-car-]\nPerceived People: [-people-]\nPerceived Bike: [-bike-]'
    # perception_str = f'There are {len(perc[0])} car and {len(perc[1])} people near the front of the car. '
    # replace_list = ['-car-', '-people-', '-bike-']
    # for k in range(3):
    #     perc_str = ''
    #     for i, p in enumerate(perc[k]):
    #         perc_str = perc_str + f'{round(p[0], 2)}, {round(p[1], 2)}; '
    #     perception_str = perception_str.replace(replace_list[k], perc_str.rstrip('; '))
    perception_str = ''
    for k in range(3):
        veh_type = 'Car' if k==0 else 'Pedestrian' if k==1 else 'Bike'
        for i, p in enumerate(perc[k]):
            if p[0]>0: lr = 'left'
            else: lr = 'right'
            if p[1]>0: fb = 'forward'
            else: fb = 'backward'
            perception_str = perception_str + f'{veh_type} {i+1}, {abs(int(p[1]))} meters {fb} and {abs(int(p[0]))} meters to the {lr};\n'

    sample = {}
    q = system.replace('-nvins-', scene['direction intention'])
    q = q.replace('-speed-', str(round(scene['speed'], 1)))
    # q = q.replace('-pr-', perception_str)
    sample['query'] = q

    # response = "Intention: [di][si]"
    # response = response.replace('di', scene['direction intention'])
    # speed_intention = 'stop' if scene['refined_intention_3']==1 else 'normal speed' if scene['refined_intention_3']==2 else 'high speed'
    # speed_intention = 'stay' if scene['refined_intention_5']==1 else 'normal speed' if scene['refined_intention_5']==2 else 'high speed' if scene['refined_intention_5']==3 else 'start up' if scene['refined_intention_5']==4 else 'brake'
    # response = response.replace('si', speed_intention)
    # if scene['target speed']==0 and scene['speed']==0:
    #     speed_intention = 'STOP'
    # elif scene['target speed']<scene['speed']-1:
    #     speed_intention = 'DECELERATE'
    # elif scene['target speed']>scene['speed']+1:
    #     speed_intention = 'ACCELERATE'
    # else:
    #     speed_intention = 'KEEP'
    if 'smoothed_speed_1s' not in scene.keys(): continue
    if scene['smoothed_speed_1s']==0:
        speed_intention = 'STOP'
    elif scene['smoothed_speed_1s']<scene['smoothed_speed']-1:
        speed_intention = 'SLOWER'
    elif scene['smoothed_speed_1s']>scene['smoothed_speed']+1:
        speed_intention = 'FASTER'
    else:
        speed_intention = 'KEEP'
    response = speed_intention

    # if intention_count[scene['refined_intention_5']-1] >= count_max[scene['refined_intention_5']-1]:
    #     continue
    # intention_count[scene['refined_intention_3']-1] += 1
    int_num = 0 if speed_intention=='STOP' else 1 if speed_intention=='SLOWER' else 2 if speed_intention=='FASTER' else 3
    if intention_count[3] >= 20000 and int_num==3: continue
    intention_count[int_num] += 1

    response = response.replace("', '", ', ').replace("['", '[').replace("']", ']')
    sample['response'] = response
    sample['images'] = scene['images']
    data.append(sample)

print(intention_count, sum(intention_count))

with open(data_file, "w") as file:
    json.dump(data, file, indent=4)