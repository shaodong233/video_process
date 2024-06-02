'''
配合多进程切分视频，并添加错误处理，因为有的视频打不开
'''

from scenedetect import detect, ContentDetector
from tqdm import tqdm
import decord
import os
import re
import json
import argparse

decord.bridge.set_bridge('native')  # Use the native bridge for Decord

def cutscene_detection(video_path, cutscene_threshold=27, max_cutscene_len=10):
    scene_list = []
    try:
        scene_list = detect(video_path, ContentDetector(threshold=cutscene_threshold, min_scene_len=15), start_in_scene=True) # 像素强度的平均变化必须超过触发剪切的阈值，阈值越高越不容易有新场景
    except Exception as e:
        print(f"Error detecting scenes in video {video_path}: {e}")
        return [], str(e)

    end_frame_idx = [0]

    try:
        video_reader = decord.VideoReader(video_path)
        fps = video_reader.get_avg_fps()
    except Exception as e:
        print(f"Failed to open video {video_path} with Decord: {e}")
        return [], str(e)

    for scene in scene_list:
        new_end_frame_idx = scene[1].get_frames()
        while (new_end_frame_idx - end_frame_idx[-1]) > (max_cutscene_len + 2) * fps:
            end_frame_idx.append(end_frame_idx[-1] + int(max_cutscene_len * fps))
        end_frame_idx.append(new_end_frame_idx)

    cutscenes = []
    for i in range(len(end_frame_idx) - 1):
        cutscenes.append([end_frame_idx[i], end_frame_idx[i + 1]])

    return cutscenes, None

def write_json_file(data, output_file):
    data = json.dumps(data, indent=4)
    def repl_func(match: re.Match):
        return " ".join(match.group().split())
    data = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, data)
    data = re.sub(r'\[\s+', '[', data)
    data = re.sub(r'],\s+\[', '], [', data)
    data = re.sub(r'\s+\]', ']', data)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cutscene Detection")
    parser.add_argument("--video-list", type=str, required=True)
    parser.add_argument("--output-json-file", type=str, default="cutscene_frame_idx.json")
    parser.add_argument("--error-json-file", type=str, default="error_cutscene_detect.json")
    parser.add_argument("-t", "--cutscene_threshold", type=int, default=25)
    args = parser.parse_args()

    # # Update output file names based on the index
    # if args.index:
    #     args.output_json_file = f"{os.path.splitext(args.output_json_file)[0]}_{args.index}.json"
    #     args.error_json_file = f"{os.path.splitext(args.error_json_file)[0]}_{args.index}.json"

    with open(args.video_list, "r", encoding="utf-8") as f:
        video_paths = f.read().splitlines()

    video_cutscenes = {}
    error_files = {}
    for video_path in tqdm(video_paths):
        cutscenes_raw, error = cutscene_detection(video_path, cutscene_threshold=args.cutscene_threshold, max_cutscene_len=5)
        if cutscenes_raw:  # Only add if cutscenes were successfully detected
            video_cutscenes[os.path.basename(video_path)] = cutscenes_raw
        elif error:  # Log the error if detection failed
            error_files[os.path.basename(video_path)] = error

    write_json_file(video_cutscenes, args.output_json_file)
    if error_files:
        write_json_file(error_files, args.error_json_file)
