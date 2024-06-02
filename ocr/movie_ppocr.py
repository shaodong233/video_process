import os
import json
from paddleocr import PaddleOCR
from decord import VideoReader, cpu
import numpy as np
import argparse
from tqdm import tqdm
import time

# 初始化OCR模型
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def process_video(video_path, output_dir, frame_interval=50):
    # 获取视频文件名，不带扩展名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 输出JSON文件路径
    output_json_path = os.path.join(output_dir, f"{video_name}.json")
    
    # 打开视频文件
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_count = len(vr)
    
    results = []

    for i in range(0, frame_count, frame_interval):
        frame = vr[i]
        image_np = frame.asnumpy()
        
        # 执行OCR识别
        result = ocr.ocr(image_np, cls=True)
        
        # 保存结果
        results.append({
            "frame_index": i,
            "ocr_result": result
        })

    # 保存OCR结果到JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def process_videos_in_directory(input_dir, output_dir, frame_interval=50):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的所有视频文件列表
    video_files = [filename for filename in os.listdir(input_dir) if filename.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    total_videos = len(video_files)

    # 初始化总的进度条
    with tqdm(total=total_videos, desc='Processing Videos', unit='video') as total_pbar:
        start_time = time.time()
        for filename in video_files:
            video_path = os.path.join(input_dir, filename)
            process_video(video_path, output_dir, frame_interval)
            total_pbar.update(1)

        elapsed_time = time.time() - start_time
        total_pbar.set_postfix({'elapsed_time': f'{elapsed_time:.2f}s'})


# process_videos_in_directory("/home/yeyang/movie_test/ocr_test", "/home/yeyang/movie_test/ocr_test")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", help="the path that save the videos", required=True)
    parser.add_argument('-o', "--output", help="the path that save the output videos", required=True)
    parser.add_argument('-f', "--frame_interval", type=int, default=50, help="the interval between frames to be processed")

    args = parser.parse_args()
    process_videos_in_directory(args.path, args.output, args.frame_interval)
