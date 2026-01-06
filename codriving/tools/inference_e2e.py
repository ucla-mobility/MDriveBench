# -*- coding: utf-8 -*-
# Author: Genjia Liu <lgj1zed@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from typing import OrderedDict
import importlib
import logging
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis_multiclass
from opencood.utils.occ_render import box2occ
torch.multiprocessing.set_sharing_strategy('file_system')

import copy

from common.io import load_config_from_yaml
from codriving.utils.torch_helper import \
    move_dict_data_to_device, build_dataloader
from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model
from common.registry import build_object_within_registry_from_config
from common.detection import warp_image
from common.torch_helper import load_checkpoint
from codriving.utils import initialize_root_logger

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("test")

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=False, default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="140.8,40",
                        help="detection range is [-140.8,+140.8m, -40m, +40m]")
    parser.add_argument('--modal', type=int, default=0,
                        help='used in heterogeneous setting, 0 lidaronly, 1 camonly, 2 ego_lidar_other_cam, 3 ego_cam_other_lidar')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--skip_frames', default=1, type=int, help="frame gap in planning data")
    parser.add_argument(
        "-c",
        "--config-file",
        default="",
        type=str,
        metavar="FILE",
        help="Config file for training",
    )
    parser.add_argument(
        "--planner_resume",
        default="",
        type=str,
        help="Path of the checkpoint from which the training resumes",
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
        default=50,
        help="Log interval",
    )
    parser.add_argument(
        "--out-dir",
        default="./out_dir/",
        type=str,
        help="directory to output model/log/etc.",
        )
    parser.add_argument(
        "--log-filename",
        default="log.txt",
        type=str,
        help="log filename",
        )
    parser.add_argument(
        "--vis-dir",
        default="vis_results/output_vis",
        type=str,
        help="directory to save visualization",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
    )
        
    
    opt = parser.parse_args()
    return opt

def visualize_output(index, model_output, batch_data, save_dir):
    from matplotlib import pyplot as plt
    output = model_output['future_waypoints']
    target = batch_data['future_waypoints']
    output_1 = output.detach().cpu().numpy()
    target_1 = target.detach().cpu().numpy()
    cmd_speed_dict = {8: 'Stop', 4: 'Slow down', 2: 'Hold', 1: 'Accelerate'}
    cmd_direction_dict = {32: 'Left', 16: 'Right', 8: 'Straight', 4: 'lane follow', 2: 'lane change left', 1: 'lane change right'}
    
    for i in range(output.shape[0]):                        
        x1, y1 = output_1[i, :, 0], output_1[i, :, 1]
        x2, y2 = target_1[i, :, 0], target_1[i, :, 1]

        y1 = np.abs(y1)
        y2 = np.abs(y2)

        plt.figure(figsize=(10, 6))
        plt.plot(x1, y1, marker='o', label='output', color='blue')
        plt.plot(x2, y2, marker='x', label='target', color='red')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        
        # try:
        # import ipdb; ipdb.set_trace()
        speed_index = int(torch.dot(batch_data["cmd_speed"][i].detach().cpu(), torch.tensor([8, 4, 2, 1], dtype=torch.float32)).item())
        direction_index = int(torch.dot(batch_data["cmd_direction"][i].detach().cpu(), torch.tensor([32, 16, 8, 4, 2, 1], dtype=torch.float32)).item())
        plt.title(f'cmd speed: {cmd_speed_dict[speed_index]}\ncmd direction: {cmd_direction_dict[direction_index]}')
        # except Exception as e:
        # import ipdb; ipdb.set_trace()
        #     plt.title(f"cmd speed: {batch_data['cmd_speed'][i]}\ncmd direction: {batch_data['cmd_direction'][i]}")


        plt.xlim(-50, 50)
        plt.ylim(0, 50)

        plt.legend()
        plt.grid()
        plt.savefig(f'{save_dir}/{index}_{i}.png')
        plt.close()

def main():
    opt = test_parser()
    log_file_name = opt.log_filename.split('.')[0]+'_'+opt.planner_resume.split('.')[0].split('_')[-1]+'.txt'
    initialize_root_logger(path=f'{opt.out_dir}/{log_file_name}')
    logging.info(f'Using perception checkpoint: {opt.model_dir}')

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']
    
    print('Creating Model')
    perception_model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, perception_model = train_utils.load_saved_model(saved_path, perception_model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        perception_model.cuda()
    perception_model.eval()

    # setting noise
    np.random.seed(30330)
    torch.manual_seed(10000)
    
    # build perception dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False, # False
                            pin_memory=False,
                            drop_last=False)
    
    if hypes['model']['args']['multi_class']:
        result_stat = {}
        class_list = [0,1,3]
        for c in class_list:
            result_stat[c] = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                    0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    
    infer_info = opt.fusion_method + opt.note

    # load planning dataset
    config = load_config_from_yaml(opt.config_file)
    data_config = config['data']
    test_data_config = data_config['test']
    test_data_config['dataset']['perception_hypes'] = hypes
    test_dataloader = build_dataloader(test_data_config, is_distributed=False)

    # build planner NN model
    model_config = config['model']
    planning_model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        model_config,
    )
    model_decoration_config = config['model_decoration']
    decorate_model(planning_model, **model_decoration_config)
    planning_model.to(device)
    last_epoch_idx = load_checkpoint(opt.planner_resume, device, planning_model, strict=True)

    # planner metric
    metric_config = config['test_metric']
    metric_func = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        metric_config,
    )

    planning_model.eval()
    if opt.vis:
        import shutil
        if os.path.exists(opt.vis_dir):
            shutil.rmtree(opt.vis_dir)
        os.makedirs(opt.vis_dir)
    ADEs = list()
    FDEs = list()    
    FVEs = list()
    AAEs = list()
    ADEs_all = list()
    FDEs_all = list()
    FVEs_all = list()
    AAEs_all = list()
    
    abnormal_dataset = dict()
    logging.info('Testing...')
    # import ipdb; ipdb.set_trace()
    print("length of test_dataloader: ", len(test_dataloader))
    # try:
    inference_time_list = []
    perc_inference_time_list = []
    for i, batch_data in enumerate(test_dataloader):
        with torch.no_grad():
            print(f"{infer_info}_{i}")

            pred_batch_data, perce_batch_data_dict = batch_data
            move_dict_data_to_device(pred_batch_data, device)

            pred_batch_data.update({'fused_feature':[],
                                    'features_before_fusion':[],})

            ############ before prediction ##########
            # perception model inference
            frame_list = list(perce_batch_data_dict.keys())
            frame_list.sort()
            perception_results_list = []
            occ_map_list = []
            for frame in frame_list:
                perce_batch_data_dict[frame] = train_utils.to_device(perce_batch_data_dict[frame], device)

                # perception_results = perception_model(perce_batch_data_dict[frame]['ego'])

                output_dict = OrderedDict()
                for cav_id, cav_content in perce_batch_data_dict[frame].items():
                    
                    perc_start_time = time.time()
                    output_dict[cav_id] = perception_model(cav_content)                    
                    perc_end_time = time.time()
                    perc_inference_time_list.append(perc_end_time - perc_start_time)
                    
                pred_box_tensor, pred_score, gt_box_tensor = \
                    test_dataloader.dataset.perception_dataset.post_process_multiclass(perce_batch_data_dict[frame],
                                        output_dict, online_eval_only=True)
                infer_result = {"pred_box_tensor" : pred_box_tensor, \
                                "pred_score" : pred_score, \
                                "gt_box_tensor" : gt_box_tensor}
                if "comm_rate" in output_dict['ego']:
                    infer_result.update({"comm_rate" : output_dict['ego']['comm_rate']})
                #################### filte ego car ###########################
                if infer_result['pred_box_tensor'][0] is not None:
                    box_filte_ego_car_list = []
                    score_filte_ego_car_list = []
                    num_car = infer_result['pred_box_tensor'][0].shape[0]
                    for car_actor_id in range(num_car):
                        car_box = copy.deepcopy(infer_result['pred_box_tensor'][0][car_actor_id])
                        car_box = car_box.cpu().numpy()
                        car_box[:,0] += 1.3
                        location_box = np.mean(car_box[:4,:2], 0)
                        if np.linalg.norm(location_box) < 1.4:
                            continue
                        box_filte_ego_car_list.append(infer_result['pred_box_tensor'][0][car_actor_id])
                        score_filte_ego_car_list.append(infer_result['pred_score'][0][car_actor_id])
                    if len(box_filte_ego_car_list) > 0:
                        infer_result['pred_box_tensor'][0] = torch.stack(box_filte_ego_car_list, dim=0)
                    else:
                        infer_result['pred_box_tensor'][0] = None
                    if len(score_filte_ego_car_list) > 0:
                        infer_result['pred_score'][0] = torch.stack(score_filte_ego_car_list, dim=0)
                    else:
                        infer_result['pred_score'][0] = None
                ##############################################

                occ_map_list.append(box2occ(infer_result))

                perception_results = output_dict['ego']
                fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2)
                fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
                w = fused_feature_2.shape[3]//2
                pred_batch_data['fused_feature'].append(fused_feature_3[:,:,:192,w-48:w+48])
                

                perception_results_list.append(perception_results)
            # warp feature in time sequence
            
            start_time = time.time()
            pred_batch_data['feature_warpped_list'] = []

            for b in range(len(perception_results_list[0]['fused_feature'])):
                feature_dim = perception_results_list[0]['fused_feature'].shape[1] # 128,256
                feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(device).float()
                det_map_pose = torch.zeros(1, 5, 3).to(device).float()
                occ_to_warp = torch.zeros(1, 5, 1, 192, 96).cuda().float()


                for t in range(5):
                    feature_to_warp[0, t, :] = pred_batch_data['fused_feature'][t][b] # occ_map_list[t]
                    det_map_pose[:, t] = torch.tensor(pred_batch_data['detmap_pose'][b,t]) # N, 3
                    occ_to_warp[0, t, 0:1] = occ_map_list[t]
                
                feature_warped = warp_image(det_map_pose, feature_to_warp)
                pred_batch_data['feature_warpped_list'].append(feature_warped)
                occ_warped = warp_image(det_map_pose, occ_to_warp)
                extra_source = {'occ':occ_warped}
                pred_batch_data['occupancy'][:,:,0,:,:] = occ_warped[:,:,0,:,:]
            ##########################################
            model_output = planning_model(pred_batch_data)
            end_time = time.time()
            inference_time_list.append(end_time - start_time)
            
            if opt.vis:
                visualize_output(i, model_output, pred_batch_data, opt.vis_dir)
                if i > 1000:    
                    break
            
            ADE, FDE, FVE, AAE = metric_func(pred_batch_data, model_output)
            abnormal_index = torch.where(ADE > 10)[0].cpu().numpy()
            try:
                for index in range(len(ADE)):
                    # import ipdb; ipdb.set_trace()
                    ADEs_all.append(ADE[[index]])
                    FDEs_all.append(FDE[[index]])
                    FVEs_all.append(FVE[[index]])
                    AAEs_all.append(AAE[[index]])
                    if index in abnormal_index:
                        # print(f"abnormal index: {index}, ADE: {ADE[index]}")
                        if pred_batch_data['scenes'][index] not in abnormal_dataset:
                            abnormal_dataset[pred_batch_data['scenes'][index]] = 1
                        else:
                            abnormal_dataset[pred_batch_data['scenes'][index]] += 1
                    else:
                        ADEs.append(ADE[[index]])
                        FDEs.append(FDE[[index]])
                        FVEs.append(FVE[[index]])
                        AAEs.append(AAE[[index]])
            except RuntimeError:
                import ipdb; ipdb.set_trace()

            torch.cuda.empty_cache()
    # except RuntimeError:
    #     print(f"RuntimeError in {i}, {len(test_dataloader)-i} left")
    #     import ipdb; ipdb.set_trace() 
    try:
        ADE_tensor = torch.cat(ADEs, dim=0)
        FDE_tensor = torch.cat(FDEs, dim=0)
        FVE_tensor = torch.cat(FVEs, dim=0)
        AAE_tensor = torch.cat(AAEs, dim=0)
        ADEs_all_tensor = torch.cat(ADEs_all, dim=0)
        FDEs_all_tensor = torch.cat(FDEs_all, dim=0)
        FVEs_all_tensor = torch.cat(FVEs_all, dim=0)
        AAEs_all_tensor = torch.cat(AAEs_all, dim=0)
    except:
        import ipdb; ipdb.set_trace()

    logging.info(f'ADE_all: {torch.mean(ADEs_all_tensor, dim=0)}, FDE_all: {torch.mean(FDEs_all_tensor, dim=0)}, FVE_all: {torch.mean(FVEs_all_tensor, dim=0)}, AAE_all: {torch.mean(AAEs_all_tensor, dim=0)}')
    logging.info(f'Exclude data with ADE > 10, ADE: {torch.mean(ADE_tensor, dim=0)}, FDE: {torch.mean(FDE_tensor, dim=0)}, FVE: {torch.mean(FVE_tensor, dim=0)}, AAE: {torch.mean(AAE_tensor, dim=0)}')
    logging.info(f'abnormal dataset: {abnormal_dataset}')
    logging.info(f'perception inference time: {np.mean(perc_inference_time_list)}')
    logging.info(f'Planner inference time: {np.mean(inference_time_list)}')
if __name__ == '__main__':
    main()
