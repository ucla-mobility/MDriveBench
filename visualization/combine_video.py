import cv2
import os
import numpy as np

# 定义目录路径
base_dir = '/GPFS/data/changxingliu/V2Xverse/results/results_driving_new_colmdriver_multiple_npc_50_0304/image/v2x_final/r23_town05_ins_oppo/r23_town05_ins_oppo_03_04_20_36_37'
output_video_path = os.path.join(base_dir, 'video.mp4')

# 确保输出目录存在
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# 获取所有 ego_vehicle_x 文件夹
ego_vehicle_folders = [f for f in os.listdir(base_dir) if f.startswith('ego_vehicle_')]
ego_vehicle_folders.sort()  # 确保按顺序处理

# 获取所有图片文件的时间帧
time_frames = sorted(set([f for folder in ego_vehicle_folders for f in os.listdir(os.path.join(base_dir, folder)) if f.endswith('.jpg')]))

# 获取第一张图片的尺寸以初始化视频
first_image_path = os.path.join(base_dir, ego_vehicle_folders[0], time_frames[0])
first_image = cv2.imread(first_image_path)
if first_image is None:
    raise ValueError(f"Failed to read first image: {first_image_path}")
height, width, _ = first_image.shape

# 压缩比例
scale_factor = 4
new_width = int(width / scale_factor)
new_height = int(height / scale_factor)

# 生成空白图（黑色）
blank_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # 压缩后的空白图

# 初始化视频写入器（使用 MPEG-4 编码器）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MPEG-4 编码器
out = cv2.VideoWriter(output_video_path, fourcc, 5.0, (new_width, new_height * len(ego_vehicle_folders)))
print(width, height * len(ego_vehicle_folders))
print(new_width, new_height * len(ego_vehicle_folders))

# 检查 VideoWriter 是否成功初始化
if not out.isOpened():
    print("Error: Failed to initialize VideoWriter")
    exit(1)

# 遍历所有时间帧
for time_frame in time_frames:
    images = []
    for folder in ego_vehicle_folders:
        image_path = os.path.join(base_dir, folder, time_frame)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                # 压缩图片
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                if len(img.shape) == 2:  # 如果是灰度图
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转换为彩色图
                images.append(img)
            else:
                print(f"Warning: Failed to read image {image_path}")
                images.append(blank_image)  # 用空白图填充
        else:
            print(f"Warning: File {image_path} does not exist")
            images.append(blank_image)  # 用空白图填充
    
    # 检查所有图片是否有效
    for img in images:
        if img is None:
            print("Error: One of the images is None")
            exit(1)
    
    # 拼接图片并写入视频
    if len(images) == len(ego_vehicle_folders):
        concatenated_image = np.vstack(images)
        out.write(concatenated_image)
    else:
        print(f"Skipping time frame {time_frame} due to missing images")

# 释放视频写入器
out.release()

# 检查视频是否成功保存
if os.path.exists(output_video_path):
    print(f"视频已成功保存到: {output_video_path}")
else:
    print("Error: 视频未成功保存")