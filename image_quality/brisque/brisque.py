'''
抽帧计算视频的BRQISUE质量评分
python brisque.py -p path/to/videos
'''

import time  
import cv2  
import torch  
from piq import BRISQUELoss  
import numpy as np  
import os  
import json
import argparse

# 仅针对固定帧率的视频
def sample_frames(video_path, sample_rate):  
    """  
    对视频进行抽帧，每隔sample_rate秒抽取一帧。  
  
    参数：  
    video_path：视频文件的路径。  
    sample_rate：抽帧的间隔，单位是秒。  
  
    返回：  
    frames：一个包含抽取的帧的列表。每一帧都是一个NumPy数组。  
    """  
    cap = cv2.VideoCapture(video_path)  
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率  
  
    frames = []  
    while True:  
        ret, frame = cap.read()  
        if not ret:  
            break  
  
        # 检查这一帧的像素值是否都一样, 避免BRISQUE计算时出现除零错误 
        if np.var(frame) > 0:  
            frames.append(frame)  
  
        # 跳过接下来的帧，以达到每隔sample_rate秒抽取一帧的目的  
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + fps * sample_rate - 1)  
  
    cap.release()  
  
    return frames

  
def main(video_dir): 
    # 初始化一个字典来保存所有的BRISQUE得分  
    brisque_scores = {}  
    # BRISQUE类实例化  
    brisque = BRISQUELoss(data_range=1.)  
    
    start = time.time()  
    
    # 遍历视频目录  
    for filename in os.listdir(video_dir):  
        # 只处理MP4文件  
        if filename.endswith('.mp4'):  
            video_path = os.path.join(video_dir, filename)  
            frames = sample_frames(video_path, sample_rate=1)  # 每隔1秒抽取一帧  
    
            video_brisque_scores = []  
            for frame in frames:  
                # 将图像转换为灰度图  
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
                # 将NumPy数组转换为PyTorch张量  
                gray = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0  

                # 检查处理后的图像的方差  
                if torch.var(gray) > 0:  
                    # 计算BRISQUE得分  
                    score = brisque(gray)  
                    # 将得分添加到列表中  
                    video_brisque_scores.append(score.item())  
    
            # 计算平均BRISQUE得分,只保留两位小数
            average_brisque_score = round(np.mean(video_brisque_scores), 2)  
    
            # 保存得分  
            brisque_scores[filename] = average_brisque_score  
    
    sorted_brisque_scores = sorted(brisque_scores.items(), key=lambda x: x[1], reverse=False)
    # 将排序结果转换回字典格式
    sorted_brisque_scores_dict = {k: v for k, v in sorted_brisque_scores} 
    # 将得分保存到JSON文件中  
    with open('brisque_scores.json', 'w') as f:  
        json.dump(sorted_brisque_scores_dict, f)  
    
    end = time.time()  
    print('Time cost: ', end - start)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='the path that save the videos', default='/Users/wangshaodong/Downloads/quality_test')

    args = parser.parse_args()
    main(args.path)
