import cv2
import os

import imageio as iio
from PIL import Image
import numpy as np

from tqdm import tqdm

def video_trans_size(input_mp4, output_h264):
    cap = cv2.VideoCapture(input_mp4)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(width,height)
    
    # 定义编解码器并创建VideoWriter对象
    out = iio.get_writer(output_h264, format='ffmpeg', mode='I', fps=10, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    while(True):
        ret, frame = cap.read()
        if ret is True:
            
            image = frame[:, :, (2, 1, 0)]
            # 写翻转的框架
            # out.write(frame)
            out.append_data(image)
            # cv.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break
    # 完成工作后释放所有内容
    cap.release()
    out.close()
    cv2.destroyAllWindows()


def concat_video(image_dir,output_path,resize_factor):
    # 图片序列的文件夹路径

    folder_a = os.path.dirname(image_dir)

    def sort_by_last_str(strings):
        # Custom sorting function that uses the last three characters to sort
        return sorted(strings, key=lambda x: int(x.split('.')[0].split('/')[-1]))

    # # Example list of strings
    # strings = ['apple', 'banana', 'cherry', 'date', 'elderberry']

    # # Sort the list
    # sorted_strings = sort_by_last_three(strings)

    # # Print the sorted list
    # print(sorted_strings)
    # 读取图片文件
    files_a = sort_by_last_str([os.path.join(folder_a, file) for file in os.listdir(folder_a) if file.endswith('.jpg') and not file.startswith('tmp')])

    # 确保两个序列长度相同
    # assert len(files_a) == len(files_b), "The sequences do not have the same number of images."

    len_min = len(files_a)
    files_a = files_a[:len_min]

    # 读取第一张图片来获取尺寸信息
    img_a = cv2.imread(files_a[0])
    height, width, layers = img_a.shape
    height = height // resize_factor
    width = width // resize_factor
    img_size = [width,height]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    video = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

    frame = 0
    # 合并图像并添加到视频
    for img_path_a in tqdm(files_a):
        frame += 1
        # if frame>200:
        #     break
        img_a = cv2.imread(img_path_a)
        
        img_a = cv2.resize(img_a,img_size)

        # 写入视频
        video.write(img_a)

    video.release()

if __name__ == '__main__':

    working_dirs = ['/GPFS/data/gjliu-1/TPAMI/V2Xverse/results/results_driving_codrivinggpt4/image/v2x_final/town05_short_collab/r320_repeat0_2_5_10/town05_short_r320_07_11_15_02_33/ego_vehicle_0'
                    ]
    
    for dir in working_dirs:
        print('process %s' % (dir))
        image_dir = os.path.join(dir,'demo')
        output_dir = os.path.join(dir.split('image')[0],'demo')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        route_name = dir.split('town05_short_collab/')[1].split('_')[0]
        h264_output_path = os.path.join(output_dir,'{}_demo_h264.mp4'.format(route_name))
        mp4_output_path = os.path.join(output_dir,'{}_demo.mp4'.format(route_name))
        concat_video(image_dir,mp4_output_path, 2)
        video_trans_size(mp4_output_path, h264_output_path)