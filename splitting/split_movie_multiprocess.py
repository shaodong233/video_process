'''
多进程切分电影：将目标文件夹下的所有电影分成num_gpus份
python split_movie_multiprocess.py -p path/to/save -o target/path/save -ct 15 -et 0.4
'''

from datetime import timedelta
import json
import os
import argparse
import subprocess
import time
from multiprocessing import Pool, cpu_count
from functools import partial

def generate_video_lists(video_dir, num_lists):
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 根据需要修改视频文件扩展名
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    chunk_size = total_files // num_lists
    remainder = total_files % num_lists

    start_idx = 0
    for i in range(num_lists):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        with open(f'video_list_{i}.txt', 'w', encoding='utf-8') as f:
            for video in video_files[start_idx:end_idx]:
                f.write(video + '\n')
        start_idx = end_idx


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

def process_video_segment(num_gpus, output, cutscene_threshold, eventcut_threshold):
    video_list_path = f'video_list_{num_gpus}.txt'
    cutscene_output = f'cutscene_frame_idx_{num_gpus}.json'
    error_cutscene_output= f'error_cutscence_detect_{num_gpus}.json'
    event_output = f'event_timecode_{num_gpus}.json'

    # 设置CUDA_VISIBLE_DEVICES环境变量  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num_gpus % 8)

    # 运行 cutscene_detect.py
    result = subprocess.run([
        'python', 'cutscene_detect_error.py', 
        '--video-list', video_list_path, 
        '--output-json-file', cutscene_output, 
        '--error-json-file', error_cutscene_output, 
        '--cutscene_threshold', str(cutscene_threshold)
    ])
    if result.returncode != 0:
        print(f"Error in cutscene_detect.py for video_list_{num_gpus}")
        return

    # 更新 video_list_path
    renew(video_list_path, cutscene_output)

    # 运行 event_stitching.py
    result = subprocess.run([
        'python', 'event_stitching.py', 
        '--video-list', video_list_path, 
        '--cutscene-frameidx', cutscene_output, 
        '--output-json-file', event_output, 
        '--eventcut_threshold', str(eventcut_threshold)
    ])
    if result.returncode != 0:
        print(f"Error in event_stitching.py for video_list_{num_gpus}")
        return

    # 运行 video_splitting.py
    result = subprocess.run([
        'python', 'video_splitting.py', 
        '--video-list', video_list_path, 
        '--event-timecode', event_output, 
        '--output-folder', output
    ])
    if result.returncode != 0:
        print(f"Error in video_splitting.py for video_list_{num_gpus}")
        return

def main(path, output, video_list_base_path='video_list', num_gpus=8, cutscene_threshold=25, eventcut_threshold=0.6):
    # 记录开始时间
    start_time = time.time()
    
    # 生成多个 video_list 文件
    generate_video_lists(path, num_gpus)

    # 并行处理每个视频分段
    process_func = partial(process_video_segment, output=output, cutscene_threshold=cutscene_threshold, eventcut_threshold=eventcut_threshold)
    with Pool(num_gpus) as pool:
        pool.map(process_func, range(num_gpus))
    # # 合并输出结果
    # combined_cutscenes = {}
    # combined_events = {}

    # for i in range(num_gpus):
    #     cutscene_output = f'cutscene_frame_idx_{i}.json'
    #     event_output = f'event_timecode_{i}.json'

    #     with open(cutscene_output, 'r', encoding='utf-8') as f:
    #         cutscenes = json.load(f)
    #         combined_cutscenes.update(cutscenes)

    #     with open(event_output, 'r', encoding='utf-8') as f:
    #         events = json.load(f)
    #         combined_events.update(events)

    # final_cutscene_output = 'cutscene_frame_idx.json'
    # final_event_output = 'event_timecode.json'

    # with open(final_cutscene_output, 'w', encoding='utf-8') as f:
    #     json.dump(combined_cutscenes, f, indent=4)

    # with open(final_event_output, 'w', encoding='utf-8') as f:
    #     json.dump(combined_events, f, indent=4)

    # # 移动所有输出视频到最终输出文件夹
    # if not os.path.exists(output):
    #     os.makedirs(output)

    # for i in range(num_gpus):
    #     output_folder = f'output_folder_{i}'
    #     for root, _, files in os.walk(output_folder):
    #         for file in files:
    #             src_file = os.path.join(root, file)
    #             dst_file = os.path.join(output, file)
    #             os.rename(src_file, dst_file)

    # 记录结束时间
    end_time = time.time()
    total_time = str(timedelta(seconds=int(end_time - start_time)))

    # 输出运行时间
    print(f"Splitting completed in {total_time}")


# main('/home/yeyang/moive_wsd/test', '/home/yeyang/moive_wsd/test_clips_multi')

if __name__ == '__main__':
    # 接受参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", help="the path that save the videos", required=True)
    parser.add_argument('-o', "--output", help="the path that save the output videos", required=True)
    parser.add_argument('-v', "--video_list_path", default='video_list.txt', help="the path that save the video list")
    parser.add_argument('-n', "--num_gpus", type=int ,default=8, help="the num of GPUs")
    parser.add_argument("-ct", "--cutscene_threshold", type=int, default=15)
    parser.add_argument('-et', "--eventcut_threshold", type=float, default=0.4)


    
    args = parser.parse_args()
    main(args.path, args.output, args.video_list_path, args.num_gpus, args.cutscene_threshold, args.eventcut_threshold)
