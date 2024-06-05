'''
计算视频的NIQE分数
python niqe_calc.py -p path/to/videos
'''
import argparse
import cv2
import numpy as np
import os
import json
import time
from module_niqe import niqe

def sample_frames(video_path, sample_rate):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if np.var(frame) > 0:
            frames.append(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + fps * sample_rate - 1)
    cap.release()
    return frames

def main(video_dir):
    niqe_scores = {}
    start = time.time()

    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            frames = sample_frames(video_path, sample_rate=1)

            video_niqe_scores = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = niqe(gray)
                video_niqe_scores.append(score)

            average_niqe_score = round(np.mean(video_niqe_scores), 2)
            niqe_scores[filename] = average_niqe_score

    sorted_niqe_scores = sorted(niqe_scores.items(), key=lambda x: x[1], reverse=False)
    # 将排序结果转换回字典格式
    sorted_niqe_scores_dict = {k: v for k, v in sorted_niqe_scores} 

    with open('niqe_scores.json', 'w') as f:
        json.dump(sorted_niqe_scores_dict, f)

    end = time.time()
    print('Time cost: ', end - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='the path that save the videos', default='/Users/wangshaodong/Downloads/quality_test')

    args = parser.parse_args()
    main(args.path)