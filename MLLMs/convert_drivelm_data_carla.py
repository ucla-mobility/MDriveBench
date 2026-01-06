# {"query": "...", "response": "...", "images": ["/GPFS/public/dataset_cop3_lidarmini/weather-0/data/routes_town10_0_w0_04_30_16_47_48/ego_vehicle_0/rgb_front/0196.jpg"]}, 
# /GPFS/data/changxingliu/DriveLM_carla/drivelm_carla_vqas      Accident        10_0        0010.json
# QA    perception / prediction / planning       [i]     Q, A
# image_paths   CAM_FRONT /    /GPFS/public/

import json
import os

data_list_new = []
event_list = ['Accident', 'DynamicObjectCrossing', 'HighwayCutIn', 'HighwayExit', 'InvadingTurn', 'MergerIntoSlowTraffic', 'NonSignalizedJunctionLeftTurn', 'NonSignalizedJunctionRightTurn', 'OppositeVehicleTakingPriority', 'ParkingCrossingPedestrian', 'PedestrianCrossing', 'PriorityAtJunction', 'SignalizedJunctionLeftTurn', 'SignalizedJunctionRightTurn', 'VehicleTurningRoute', 'VehicleTurningRoutePedestrian', 'EnterActorFlow', 'AccidentTwoWays', 'StaticCutIn', 'EnterActorFlowV2']

base_path = '/GPFS/data/changxingliu/DriveLM_carla/drivelm_carla_vqas'
for event in os.listdir(base_path):
    if event not in event_list: continue
    event_path = os.path.join(base_path, event)
    print(event)
    for route in os.listdir(event_path):
        route_path = os.path.join(event_path, route)
        for file_name in os.listdir(route_path):
            file_path = os.path.join(route_path, file_name)

            with open(file_path, 'r') as file:
                data_list = json.load(file)

            images = []
            for cam, img in data_list['image_paths'].items():
                if img != None:
                    images.append(f'/GPFS/public/{img}')

            for key, qa_list in data_list['QA'].items():
                for qa in qa_list:
                    data_new = {
                        'query': qa['Q'],
                        'response': qa['A'],
                        'images': images
                    }
                    data_list_new.append(data_new)
                    
print(len(data_list_new))

# output_file_path = '/GPFS/data/changxingliu/InternVL/data/data_drivelm_carla.json'
# with open(output_file_path, 'w') as output_file:
#     json.dump(data_list_new, output_file, indent=4)

output_file_path = '/GPFS/data/changxingliu/InternVL/data/data_drivelm_carla_200k.json'
import random
random.shuffle(data_list_new)
with open(output_file_path, 'w') as output_file:
    json.dump(data_list_new[:200000], output_file, indent=4)