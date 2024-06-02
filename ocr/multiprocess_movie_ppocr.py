import os
import json
from paddleocr import PaddleOCR
from decord import VideoReader, cpu
import numpy as np
import argparse
from tqdm import tqdm
import time
from multiprocessing import Pool

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

def process_videos(video_files, output_dir, frame_interval=50, num_gpus=0): 
    # 设置CUDA_VISIBLE_DEVICES环境变量  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num_gpus % 8)
      
    total_videos = len(video_files)  
  
    # 初始化总的进度条  
    with tqdm(total=total_videos, desc='Processing Videos', unit='video') as total_pbar:  
        start_time = time.time()  
        for filename in video_files:  
            process_video(filename, output_dir, frame_interval)  
            total_pbar.update(1)  
  
        elapsed_time = time.time() - start_time  
        total_pbar.set_postfix({'elapsed_time': f'{elapsed_time:.2f}s'})  
  
def divide_files(files, n):  
    avg = len(files) // n  
    remainder = len(files) % n  
    div_points = [avg * i + min(i, remainder) for i in range(n + 1)]  
    return [files[div_points[i]:div_points[i + 1]] for i in range(n)]  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument('-p', "--path", help="the path that save the videos", required=True)  
    parser.add_argument('-o', "--output", help="the path that save the output videos", required=True)  
    parser.add_argument('-f', "--frame_interval", type=int, default=50, help="the interval between frames to be processed")  
    parser.add_argument('-n', "--num_gpus", type=int, default=8, help="the number of GPUs")  
  
    args = parser.parse_args()  
    
    # 确保输出目录存在  
    if not os.path.exists(args.output):  
        os.makedirs(args.output) 

    # 获取输入目录中的所有视频文件列表  
    video_files = [os.path.join(args.path, filename) for filename in os.listdir(args.path) if filename.endswith(('.mp4', '.avi', '.mov', '.mkv'))]  
  
    # 平均分配视频文件  
    divided_files = divide_files(video_files, args.num_gpus)  
  
    # 创建一个进程池  
    with Pool(args.num_gpus) as p:  
    # 并行处理视频  
        p.starmap(process_videos, [(files, args.output, args.frame_interval, i) for i, files in enumerate(divided_files)]) 