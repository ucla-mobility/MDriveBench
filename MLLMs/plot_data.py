import json
import os
import matplotlib.pyplot as plt
import numpy as np


json_file_path = '/GPFS/data/changxingliu/InternVL/data/data_sum_1003_speed_level_5.json'
with open(json_file_path, 'r') as f:
    data = json.load(f)

save_path = '/GPFS/data/changxingliu/InternVL/data/plot_data'
os.makedirs(save_path, exist_ok=True)

intentions_map = {
    'start': {'start up': 1},
    'stay': {'stay still while danger': 1},
    'stop': {'stop from moving for danger': 1},
    'normal': {'target speed': 4},
    'fast': {'target speed': 6}
}

scene_data = {}
for record in data:
    image_path = record['images'][0]
    scene = image_path.split('/')[-4]
    frame = image_path.split('/')[-1].split('.')[0]
    
    # get intention
    for intention, conditions in intentions_map.items():
        if all(record.get(condition_key) == condition_value for condition_key, condition_value in conditions.items()):
            break
    
    # update scene data
    if scene not in scene_data:
        scene_data[scene] = {'frames': [], 'intention': [], 'speed': []}
    
    scene_data[scene]['frames'].append(int(frame))
    scene_data[scene]['intention'].append(intention)
    scene_data[scene]['speed'].append(record['speed'])

# plot intention and speed relation
for scene, data in scene_data.items():
    frames = data['frames']
    intentions = data['intention']
    speeds = data['speed']
    
    intention_to_num = {'start': 1, 'stay': 2, 'stop': 3, 'normal': 4, 'fast': 5}
    num_intentions = [intention_to_num[intention] for intention in intentions]

    # smooth the speed data
    window_length = 5 if len(frames) > 20 else 5 # smooth window len
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
        if i+3>len(filtered_intention)-1:
            adjust_intention.append(filtered_intention[-1])
        else:
            adjust_intention.append(filtered_intention[i+3])

    level5_intention = []
    for i, (ori_i, adj_i) in enumerate(zip(filtered_intention, adjust_intention)):
        if i<10 and ori_i == 1:
            level5_intention.append(0) # start
        elif ori_i == 1 and adj_i != 1:
            level5_intention.append(0)
        elif ori_i != 1 and adj_i == 1:
            level5_intention.append(2) # brake
        elif adj_i == 1:
            level5_intention.append(1)
        elif adj_i == 2:
            level5_intention.append(3)
        elif adj_i == 3:
            level5_intention.append(4)
        
    # create subplots
    length = int(len(frames)/10) if len(frames) > 200 else 20
    fig, axs = plt.subplots(4, 1, figsize=(length, 16))
    
    # plot intention
    axs[0].plot(frames, num_intentions, marker='.')
    axs[0].set_title(f'Intention Changes Over Time for Scene: {scene}')
    axs[0].set_xlabel('Frame Number')
    axs[0].set_ylabel('Intention Value')
    axs[0].set_yticks([1, 2, 3, 4, 5])
    axs[0].set_yticklabels(['start', 'stay', 'stop', 'normal', 'fast'])
    axs[0].grid(True)
    
    # plot speed
    axs[1].plot(frames, speeds, label='Original Speed', marker='.')
    axs[1].plot(frames, smoothed_speeds, label='Smoothed Speed', marker='.', linestyle='--')
    axs[1].set_title(f'Speed Changes Over Time for Scene: {scene}')
    axs[1].set_xlabel('Frame Number')
    axs[1].set_ylabel('Speed')
    axs[1].grid(True)

    axs[2].plot(frames, intention_from_speed, marker='.')
    axs[2].set_title(f'Intention Extracted from Speed for Scene: {scene}')
    axs[2].set_xlabel('Frame Number')
    axs[2].set_ylabel('Intention Value')
    axs[2].set_yticks([1, 2, 3])
    axs[2].set_yticklabels(['stop', 'normal', 'fast'])
    axs[2].grid(True)
    
    axs[3].plot(frames, filtered_intention, label='filtered intention', marker='.')
    axs[3].plot(frames, smoothed_speeds, label='Smoothed Speed', marker='.', linestyle='--')
    axs[3].plot(frames, level5_intention, label='level5_intention', marker='.', linestyle='--')
    axs[3].set_title(f'Intention for Scene: {scene}')
    axs[3].set_xlabel('Frame Number')
    axs[3].set_ylabel('Intention Value')
    axs[3].set_yticks([1, 2, 3])
    axs[3].set_yticklabels(['stop', 'normal', 'fast'])
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{scene}.png'))
    plt.close()

print("All combined plots processed.")