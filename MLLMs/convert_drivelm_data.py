# /GPFS/data/changxingliu/V2Xverse/DriveLM/challenge/test_llama.json
# {"query": "...", "response": "...", "images": ["/GPFS/public/dataset_cop3_lidarmini/weather-0/data/routes_town10_0_w0_04_30_16_47_48/ego_vehicle_0/rgb_front/0196.jpg"]}, 
''' {
        "id": "f0f120e4d4b0441da90ec53b16ee169d_4a0798f849ca477ab18009c3a20b7df2_0",
        "image": [
            "data/nuscenes/samples/CAM_FRONT/n008-2018-09-18-13-10-39-0400__CAM_FRONT__1537291010612404.jpg",
            "data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_LEFT__1537291010604799.jpg",
            "data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_FRONT_RIGHT__1537291010620482.jpg",
            "data/nuscenes/samples/CAM_BACK/n008-2018-09-18-13-10-39-0400__CAM_BACK__1537291010637558.jpg",
            "data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-09-18-13-10-39-0400__CAM_BACK_LEFT__1537291010647405.jpg",
            "data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-09-18-13-10-39-0400__CAM_BACK_RIGHT__1537291010628113.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."
            },
            {
                "from": "gpt",
                "value": "There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,1088.3,497.5>, <c2,CAM_BACK,864.2,468.3>, and <c3,CAM_FRONT,1043.2,82.2>."
            }
        ]
    },'''

import json

file_path = '/GPFS/data/changxingliu/DriveLM/challenge/test_llama.json'
with open(file_path, 'r') as file:
    data_list = json.load(file)

data_list_new = []
for data in data_list:
    data_new = {
        'query': data['conversations'][0]['value'].replace('<image>\n', ''),
        'response': data['conversations'][1]['value'],
        'images': data['image']
    }
    data_list_new.append(data_new)

output_file_path = '/GPFS/data/changxingliu/InternVL/data/data_drivelm.json'
with open(output_file_path, 'w') as output_file:
    json.dump(data_list_new, output_file, indent=4)