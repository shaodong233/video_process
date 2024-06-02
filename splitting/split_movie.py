'''
单进程切分视频
'''

from datetime import timedelta
import json
import os
import argparse
import subprocess
import time  

# 生成video_list.txt
def generate_video_list(video_dir, video_list_path):
    with open(video_list_path, 'w', encoding='utf-8') as f:
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 根据需要修改视频文件扩展名
                    f.write(os.path.join(root, file) + '\n')

def renew(video_list_path, cutscene_output):
    with open(video_list_path, 'r', encoding='utf-8') as f:
        video_list = f.readlines()

    with open(cutscene_output, 'r', encoding='utf-8') as f:
        cutscenes = json.load(f)

    video_names = set(cutscenes.keys())

    with open(video_list_path, 'w', encoding='utf-8') as f:
        for video_path in video_list:
            video_name = os.path.basename(video_path.strip())
            if video_name in video_names:
                f.write(video_path)

def main(path, output, video_list_path='video_list.txt'):
    video_list_path = video_list_path
    cutscene_output = 'cutscene_frame_idx.json'
    event_output = 'event_timecode.json'

    # 记录开始时间
    start_time = time.time()
    
    generate_video_list(path, video_list_path)

    # rename_files_and_folders(args.path, video_list_path)
    
    # 调用 cutscene_detect.py 脚本
    # 找过场动画
    result = subprocess.run(['python', 'custcene_detect_error.py', '--video-list', video_list_path, '--output-json-file', cutscene_output])
    if result.returncode != 0:
        print("Error in cutscene_detect.py")
        return
    
    # 更新video_list_path
    renew(video_list_path, cutscene_output)

    # 合起来
    result = subprocess.run(['python', 'event_stitching.py', '--video-list', video_list_path, '--cutscene-frameidx', cutscene_output, '--output-json-file', event_output])
    if result.returncode != 0:
        print("Error in event_stitching.py")
        return
    # 切开
    result = subprocess.run(['python', 'video_splitting.py', '--video-list', video_list_path, '--event-timecode', event_output, '--output-folder', output])
    if result.returncode != 0:
        print("Error in video_splitting.py")
        return

     # 记录结束时间
    end_time = time.time()
    total_time = str(timedelta(seconds=int(end_time - start_time)))

    # 输出运行时间
    print(f"Splitting completed in {total_time} seconds.")

# main('/home/yeyang/moive_wsd/test', '/home/yeyang/moive_wsd/test_clips')

if __name__ == '__main__':
    # 接受参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", help="the path that save the videos")
    parser.add_argument('-o', "--output", help="the path that save the output videos")
    parser.add_argument('-v', "--video_list_path", default='video_list.txt', help="the path that save the video list")

    args = parser.parse_args()
    main(args.path, args.output, args.video_list_path)
