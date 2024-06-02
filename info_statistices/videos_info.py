'''
多进程统计目标目录下所有视频的时长、分辨率、帧数和宽高比，并将结果保存为json文件，同时绘制分辨率和宽高比的分布图。
python videos_info.py -i /remote-home1/dataset/mixkit -o video_info.json -p1 video_resolution_plot.png -p2 video_aspect_ratio_plot.png
'''

import os  
import json  
import subprocess  
from multiprocessing import Pool, cpu_count  
import argparse  
import matplotlib.pyplot as plt  
from collections import defaultdict  
import math  
  
def get_video_info(video_path):  
    cmd_duration = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "' + video_path + '"'  
    cmd_resolution = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "' + video_path + '"'  
    cmd_frames = 'ffprobe -v error -select_streams v -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "' + video_path + '"'  
  
    try:  
        duration = float(subprocess.check_output(cmd_duration, shell=True).decode('utf-8').strip())  
        resolution = subprocess.check_output(cmd_resolution, shell=True).decode('utf-8').strip()  
        frames = int(subprocess.check_output(cmd_frames, shell=True).decode('utf-8').strip())  
        width, height = map(int, resolution.split('x'))  
        gcd = math.gcd(width, height)  
        aspect_ratio = f'{width//gcd}:{height//gcd}'  # calculate aspect ratio and simplify it  
    except Exception:  # Catch all exceptions  
        duration = 'error'  
        resolution = 'error'  
        frames = 'error'  
        aspect_ratio = 'error'  
  
    return video_path, {'duration': duration, 'resolution': resolution, 'frames': frames, 'aspect_ratio': aspect_ratio}  
  
def plot_distribution(data, plot_path, xlabel, total_num):  
    counter = defaultdict(int)  
    for item in data:  
        counter[item] += 1  
    
    threshold = 0.01 * total_num
    # Filter out items with counts less than threshold  
    counter = {k: v for k, v in counter.items() if v >= threshold}  
  
    items, counts = zip(*counter.items())  
    fig, ax = plt.subplots()  
    bars = ax.bar(items, counts)  
  
    for bar in bars:  
        yval = bar.get_height()  
        ax.text(bar.get_x() + bar.get_width() / 2, yval, yval, ha='center', va='bottom')  
  
    plt.xlabel(xlabel)  
    plt.ylabel('Count')  
    plt.xticks(rotation=45)  # Change rotation to 45 degrees  
    plt.tight_layout()  # This will ensure that all labels are visible  
    plt.savefig(plot_path)  
  
def main(args):  
    video_formats = ['.mp4', '.avi', '.flv', '.mov', '.wmv', '.mkv']  
    video_files = [os.path.join(root, f)  
                   for root, dirs, files in os.walk(args.video_dir)  
                   for f in files if any(f.endswith(format) for format in video_formats)]  
  
    print(f'Total number of video files: {len(video_files)}')  
  
    with Pool(cpu_count()) as p:  
        result = dict(p.map(get_video_info, video_files))  
  
    total_duration = sum(info['duration'] for info in result.values() if info['duration'] != 'error')  
    total_frames = sum(info['frames'] for info in result.values() if info['frames'] != 'error')  
    error_count = sum(1 for info in result.values() if info['duration'] == 'error' or info['frames'] == 'error' or info['aspect_ratio'] == 'error')  # Count the number of errors  
    
    total_duration = str(round(total_duration / 3600, 2)) + ' hours' if total_duration >= 3600 else str(round(total_duration / 60, 2)) + ' minutes' if total_duration >= 60 else str(total_duration) + ' seconds'
    
    print(f'Total duration: {total_duration}')  
    print(f'Total frames: {total_frames}')  
    print(f'Number of files with errors: {error_count}')  

    # 先将total_duration，total_frames，error_count写入json文件
    result['total_duration'] = total_duration
    result['total_frames'] = total_frames
    result['error_count'] = error_count

    with open(args.json_path, 'w') as f:  
        json.dump(result, f, indent=4)  
  
    resolutions = [info['resolution'] for info in result.values() if info['resolution'] != 'error']  
    aspect_ratios = [info['aspect_ratio'] for info in result.values() if info['aspect_ratio'] != 'error']  
  
    plot_distribution(resolutions, args.plot_path_resolution, 'Resolution', len(video_files))  
    plot_distribution(aspect_ratios, args.plot_path_aspect_ratio, 'Aspect Ratio', len(video_files))  
  
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument('-i', '--video_dir', default='/remote-home1/dataset/mixkit', help='Path to the directory of video files.')  
    parser.add_argument('-o', '--json_path', default='video_info.json', help='Path to save the json file.')  
    parser.add_argument('-p1', '--plot_path_resolution', default='video_resolution_plot.png', help='Path to save the resolution plot.')  
    parser.add_argument('-p2', '--plot_path_aspect_ratio', default='video_aspect_ratio_plot.png', help='Path to save the aspect ratio plot.')  
    args = parser.parse_args()  
  
    main(args)  