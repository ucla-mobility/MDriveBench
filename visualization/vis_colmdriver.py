import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

import numpy as np
import math

def project_points_to_image(world_points, R, T, K):
    """
    将世界坐标系下的点投影到相机图像上

    :param world_points: 世界坐标系中的点 (Nx3)
    :param R: 相机旋转矩阵 (3x3)
    :param T: 相机平移矩阵 (3x1)
    :param K: 相机内参矩阵 (3x3)
    :return: 图像坐标系下的点 (Nx2)
    """
    # 将世界坐标系的点转换到相机坐标系下
    points_camera = np.dot(R, world_points.T).T + T.T
    
    # 将相机坐标系的点投影到图像平面
    points_image = np.dot(K, points_camera.T).T
    
    # 进行归一化
    points_image = points_image[:, :2] / points_image[:, 2:3]
    
    return points_image

def project_points_to_image_2(waypoints_veh, camera_rel_pos, K):
    """
    将世界坐标系下的点投影到相机图像上

    :param world_points: 世界坐标系中的点 (Nx3)
    :param R: 相机旋转矩阵 (3x3)
    :param T: 相机平移矩阵 (3x1)
    :param K: 相机内参矩阵 (3x3)
    :return: 图像坐标系下的点 (Nx2)
    """
    # 将世界坐标系的点转换到相机坐标系下
    z = camera_rel_pos[2] * np.ones_like(waypoints_veh[:, 0:1])
    points_camera = np.concatenate([waypoints_veh, z], axis=1)
    
    # 将相机坐标系的点投影到图像平面
    points_image = np.dot(K, points_camera.T).T
    
    # 进行归一化
    points_image = points_image[:, :2] / points_image[:, 2:3]
    
    return points_image


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角（roll, pitch, yaw）转换为旋转矩阵
    """
    # 计算每个旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    # 总旋转矩阵是三个旋转矩阵的乘积
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def camera_to_world(ego_pos, ego_yaw, camera_rel_pos, camera_euler_angles):
    """
    计算相机在世界坐标系下的外参
    :param ego_pos: 自车在世界坐标系的位置 (x_ego, y_ego, z_ego)
    :param ego_yaw: 自车的yaw角度
    :param camera_rel_pos: 相机相对于自车的位置 (x_cam, y_cam, z_cam)
    :param camera_euler_angles: 相机的欧拉角 (roll, pitch, yaw)
    :return: 相机在世界坐标系下的位置和姿态
    """
    # 计算自车到世界坐标系的旋转矩阵
    R_ego_to_world = euler_to_rotation_matrix(0, 0, ego_yaw)
    
    # 计算相机相对于自车的旋转矩阵
    R_cam_to_ego = euler_to_rotation_matrix(camera_euler_angles[0], camera_euler_angles[1], camera_euler_angles[2])
    
    # 相机在世界坐标系的旋转矩阵
    R_cam_to_world = np.dot(R_ego_to_world, R_cam_to_ego)
    
    # 相机在世界坐标系的平移向量
    camera_world_pos = np.dot(R_ego_to_world, camera_rel_pos) + ego_pos
    
    return camera_world_pos, R_cam_to_world

def project_points_to_vehicle(world_points, ego_pos, ego_yaw):
    """
    将世界坐标系下的点投影到自车坐标系下
    :param world_points: 世界坐标系中的点 (Nx2)
    :param ego_pos: 自车在世界坐标系的位置 (x_ego, y_ego)
    :param ego_yaw: 自车的yaw角度
    :return: 自车坐标系下的点 (Nx2)
    """
    # 计算自车到世界坐标系的旋转矩阵
    R = np.array([[math.cos(ego_yaw), -math.sin(ego_yaw)],
                  [math.sin(ego_yaw), math.cos(ego_yaw)]])
    world_points = world_points - ego_pos[:2]
    vehicle_points = np.dot(R.T, world_points.T).T
    return vehicle_points

def draw_text_with_wrap(image, text, position, font, font_scale, color, thickness, max_width,text_y):
    """
    在图像上绘制自动换行的文本。
    
    :param image: 要绘制文本的图像
    :param text: 要绘制的文本
    :param position: 文本起始位置 (x, y)
    :param font: 字体类型
    :param font_scale: 字体大小
    :param color: 字体颜色
    :param thickness: 字体厚度
    :param max_width: 每行文本的最大宽度（控制换行）
    """
    # 将文本按最大宽度拆分成多行
    lines = []
    words = text.split(' ')
    current_line = ""

    for word in words:
        # 检查添加当前单词后是否超过最大宽度
        test_line = current_line + ' ' + word if current_line else word
        test_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        
        if test_size[0] <= max_width:
            current_line = test_line  # 当前行可以容纳这个单词
        else:
            lines.append(current_line)  # 添加当前行到结果
            current_line = word  # 重新开始新的一行

    lines.append(current_line)  # 添加最后一行

    # 将每行文本绘制到图像上
    y_position = position[1]
    for line in lines:
        cv2.putText(image, line, (position[0], y_position), font, font_scale, color, thickness)
        y_position += 100  # 设置每行的垂直间隔
        text_y+=100
    return text_y

def draw_world_points(bev_image,waypoints,ego_pos,ego_yaw,camera_rel_pos,K,color=(100, 100, 255),r=10):
    waypoints_veh = project_points_to_vehicle(waypoints,ego_pos,ego_yaw)
    image_points = project_points_to_image_2(waypoints_veh, camera_rel_pos, K)
    for point in image_points:
        cv2.circle(bev_image, (int(point[0]), int(point[1])), r, color, -1)
    return bev_image

def images_to_video(image_folder, video_name, fps=30, size=(640, 480)):
    # 获取图片文件夹中的所有图片
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg") and 'demo' in img]
    images.sort()  # 如果图片文件名有序，可以根据文件名排序

    # 使用 VideoWriter 创建视频文件
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编解码器（XVID 或 MJPG）
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    for image_name in images:
        # 读取图片
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)

        # 调整图片大小（如果需要）
        img_resized = cv2.resize(img, size)

        # 写入视频
        video.write(img_resized)

    # 释放 VideoWriter 对象
    video.release()
    print(f"视频已保存为 {video_name}")

def demo_from_results(results_dir):
    # results_dir = '/GPFS/data/gjliu-1/TPAMI/V2Xverse/results/results_driving_colmdriver_nego_repeat_3_double/image/v2x_final/town05_short_r1_ins_c'
    skip_frame = 4
    ego_id = '0'
    demo_width = 5500
    demo_height = 1500

    img_dir = os.path.join(results_dir,sorted(os.listdir(results_dir))[-1],f'ego_vehicle_{ego_id}')
    log_dir = os.path.join(results_dir,sorted(os.listdir(results_dir))[-1],'log')
    nego_log = json.load(open(os.path.join(log_dir, 'nego.json')))
    json_list = sorted([x for x in os.listdir(img_dir) if x.endswith('.json')])
    max_frame = int(json_list[-1].split('.')[0])

    camera_rel_pos = np.array([0, 0, 20])  # 相机相对于自车的位置 (x_cam, y_cam, z_cam)
    camera_euler_angles = np.array([0, 0, math.radians(90)])  # 相机的roll, pitch, yaw角度
    # 计算相机在世界坐标系下的外参
    K = np.array([[629.32472338296, 0.0, 750.0],
                    [0.0, 629.32472338296, 750.0],
                    [0.0, 0.0, 1.0]])

    # waypoints_color_list = [(255,200,200),(150,255,150),(100,100,255)]
    waypoints_color_list = [(255,0,0),(0,255,0),(0,0,255)]

    for frame in tqdm(range(0, max_frame, skip_frame)):
        # if frame > 50:
        #     break
        demo_image = np.zeros((demo_height, demo_width, 3), dtype=np.uint8)
        bev_image = cv2.imread(os.path.join(img_dir, f'{frame:04d}_bev.jpg'))
        bev_height, bev_width = bev_image.shape[:2]
        driving_states = json.load(open(os.path.join(img_dir, f'{frame:04d}.json')))
        nego_list = nego_log['content'][str(frame)]
        last_analysis = 'None'
        last_intention = 'None'
        for nego in nego_list:
            if nego[0] == nego[1] and str(nego[0]) == ego_id:
                last_analysis = 'analysis: '+ nego[2].split('Brief analysis: ')[-1].split('\n')[0]
                last_intention = 'intention: '+ nego[2].split('Intention: ')[-1].split('\n')[0]
        id_text = f'veh_id: {ego_id}'
        speed_text = f'speed: {driving_states["speed"]:04f}m/s'
        throttle_text = f'throttle: {driving_states["throttle"]:04f}'
        brake_text = f'brake: {driving_states["brake"]:04f}'
        steer_text = f'steer: {driving_states["steer"]:04f}'
        frame_text = f'frame: {frame:04d}'

        # compute camera config
        ego_pos = np.array([driving_states['x'], driving_states['y'], 0])  # 自车在世界坐标系的位置 (x_ego, y_ego, z_ego)
        ego_yaw = driving_states['theta'] - np.pi*0/2  # 自车的yaw角度

        nego_times = len(nego_log['waypoints'][str(frame)])
        bev_y = 0
        bev_x = 1500

        cv2.circle(bev_image, (150,100), 20, (255,255,255), -1)
        cv2.putText(bev_image, f'target point', (300,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
        # draw waypoints
        for nego_id in range(nego_times):
            color = waypoints_color_list[nego_id + len(waypoints_color_list)-nego_times]
            waypoints = np.array(nego_log['waypoints'][str(frame)][nego_id])
            waypoints = waypoints.reshape(-1, 2)
            waypoints[:,0] = waypoints[:,0] + (nego_times-nego_id-1)*0.2
            waypoints[:,1] = waypoints[:,1] + (nego_times-nego_id-1)*0.2
            bev_image = draw_world_points(bev_image,waypoints,ego_pos,ego_yaw,camera_rel_pos,K,color,r=5)
            cv2.line(bev_image, (100, 100+80*(nego_id+1)), (200, 100+80*(nego_id+1)), color, 5)
            cv2.putText(bev_image, f'negotiation {nego_id}', (300, 100+80*(nego_id+1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)


        target_point = driving_states['target']
        target_point = np.array([target_point])
        bev_image = draw_world_points(bev_image,target_point,[0,0,0],0,camera_rel_pos,K,(255,255,255),r=20)

        

        demo_image[bev_y:bev_y+bev_height, bev_x:bev_x+bev_width] = bev_image
        h_line_y_0 = bev_height
        v_line_x_0 = bev_x
        v_line_x_1 = bev_x+bev_width
        cv2.line(demo_image, (0, h_line_y_0), (demo_image.shape[1], h_line_y_0), (255, 255, 255), 20)
        cv2.line(demo_image, (v_line_x_0, 0), (v_line_x_0, h_line_y_0), (255, 255, 255), 20)
        cv2.line(demo_image, (v_line_x_1, 0), (v_line_x_1, h_line_y_0), (255, 255, 255), 20)

        text_y_0 = 100
        for text in [id_text,frame_text,speed_text,throttle_text,brake_text,steer_text,last_intention,last_analysis]:
            # cv2.putText(demo_image, text, (50,text_y_0), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
            draw_text_with_wrap(demo_image, text, (50,text_y_0), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6,1400,text_y_0)
            text_y_0 += 100
        text_y_1 = 100
        for i,nego in enumerate(nego_list):
            if nego[0] != nego[1]:
                text = f'veh{nego[0]}->veh{nego[1]}:{nego[2]}'
                text_y_1 = draw_text_with_wrap(demo_image, text, (bev_x+bev_width+50,text_y_1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6,2400,text_y_1)
            if i>0 and (i+1)%4==0:
                safety_score = nego_log['safety_score'][str(frame)][i//4]
                efficiency_score = nego_log['efficiency_score'][str(frame)][i//4]
                cons_score = nego_log['cons_score'][str(frame)][i//4]
                score_text = f'safety: {safety_score:.2f}/1,efficiency: {efficiency_score:.2f}/1, consensus: {cons_score:.2f}/1'
                text_y_1 = draw_text_with_wrap(demo_image, score_text, (bev_x+bev_width+50,text_y_1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 255), 6,2400,text_y_1)
        # 保存为JPG文件
        cv2.imwrite(os.path.join(img_dir,f'{frame:04d}_demo.jpg'), demo_image)
    # 保存为视频
    video_name = os.path.join(log_dir,'demo.mp4')
    images_to_video(img_dir, video_name, fps=5, size=(demo_width, demo_height))



for dir in [
            '/GPFS/data/changxingliu/V2Xverse/results/results_driving_intersection_colmdriver_triple_0214/image/v2x_final/town05_short_r1_ins_sl',
            '/GPFS/data/changxingliu/V2Xverse/results/results_driving_intersection_colmdriver_triple_0214/image/v2x_final/town05_short_r2_ins_sr'
            ]:
    demo_from_results(dir)