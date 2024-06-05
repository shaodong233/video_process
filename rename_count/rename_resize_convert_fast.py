# 重命名所有文件夹为uuid，重命名所有文件为uuid
# 将所有视频文件转换为mp4格式，4K、2K等视频分辨率调整为1080p，视频质量调整为CRF 23，视频编码速度调整为fast，删除原视频文件
# 生成info.json文件记录时长和错误信息

# python -p path/to/save -o path/to/output -n bbc05 > bbc05.rename.log 2>&1

import os
import json
import concurrent.futures
import shutil
import uuid
import decord
import numpy as np
from datetime import timedelta
import time
import subprocess
import argparse

# 获取视频宽和高
def get_video_dimensions(path):
    try:
        result = subprocess.run(
            ['ffprobe', '-hide_banner', '-loglevel', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        output = result.stdout.strip()
        if output:
            w, h = map(int, output.split(','))
            return w, h
        else:
            print(f"ffprobe did not return any dimensions for {path}")
            return None, None
    except Exception as e:
        print(f"Error getting dimensions for {path}: {e}")
        return None, None

def process_movie(old_path, final_path):
    print(f"Processing {old_path}")
    try:
        w, h = get_video_dimensions(old_path)
        if w is None or h is None:
            os.remove(old_path)
            return old_path, None

        # 最短边大于1080时，resize小到1080
        if min(w, h) > 1080:
            scale_filter = f'scale=-2:1080' if w > h else f'scale=1080:-2'
        else:
            # 最短边小于等于1080时，保持原分辨率
            scale_filter = None

        if scale_filter:
            # -vf scale_filter: 使用视频过滤器（scale_filter）调整视频分辨率。
            # -sn: 禁用字幕流。
            # -c:v libx264: 使用 libx264 编码器编码视频流。
            # -an: 禁用音频流。
            # new_path: 指定输出文件路径。
            # -y: 覆盖输出文件（若存在）
            # '-preset', 'fast'：使用 fast 预设，速度和压缩效率的平衡。其他预设包括 ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow。
            # '-crf', '23'：设置恒定质量的 CRF 值为 23。值越小质量越好（但文件越大），值越大质量越差（但文件越小）。一般推荐范围在 18 到 23 之间。
            # '-threads', '4'（可选）：指定使用 4 个线程（根据你的 CPU 核心数量调整）。可以增加编码速度。
            command = [
                'ffmpeg', '-i', old_path, '-vf', scale_filter, '-sn', '-c:v', 'libx264',
                '-preset', 'fast', '-crf', '23', '-an', final_path, '-y'
            ]
            subprocess.run(command, check=True)

        # 不需要resize的mp4文件不进行ffmpeg处理
        elif old_path.endswith('.mp4'):
            # 如果final_path和old_path不是同一个目录时，可以直接复制文件进行重定向
            if old_path != final_path:
                # shutil.move(old_path, final_path) # windows
                command = ['mv', old_path, final_path] # linux
                subprocess.run(command, check=True)
            else: 
                # 同一个目录时直接返回
                vr = decord.VideoReader(old_path)
                fps = vr.get_avg_fps()
                duration = len(vr) / fps
                print(f"Skipped processing {old_path}. Duration: {duration}")
                return old_path, duration
        else:
            # 不需要resize的其他视频文件，直接转换成mp4格式
            command = [
                'ffmpeg', '-i', old_path, '-c', 'copy', '-y', final_path
            ]
            subprocess.run(command, check=True)

        # 防止原文件移动
        if os.path.exists(old_path):
            os.remove(old_path)
        print(f"Converted and deleted {old_path}")

        vr = decord.VideoReader(final_path)
        fps = vr.get_avg_fps()
        duration = len(vr) / fps
    except Exception as e:
        print(f"Error while processing {old_path}: {e}")
        return old_path, None

    print(f"Finished processing {final_path}. Duration of videos: {duration}")
    return final_path, duration

def rename_folder_to_uuid(folder_path):
    parent_dir = os.path.dirname(folder_path)
    new_folder_name = str(uuid.uuid4())
    new_folder_path = os.path.join(parent_dir, new_folder_name)
    os.rename(folder_path, new_folder_path)
    return new_folder_path

def rename_all_folders(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            old_dir_path = os.path.join(root, dir_name)
            new_dir_path = rename_folder_to_uuid(old_dir_path)
            print(f"Renamed folder {old_dir_path} to {new_dir_path}")

def process_folder(folder_path, output_path=None, info_json_name='default'):
    error_list = []
    start_time = time.time()

    rename_all_folders(folder_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:                
                old_file_path = os.path.join(root, file)

                # 忽略系统文件夹
                if old_file_path.lower().startswith("e:\\$recycle.bin"):
                    continue

                if file.endswith('.pdf'):
                    new_file = file[:-4]
                    new_file_path = os.path.join(root, new_file)
                    os.rename(old_file_path, new_file_path)
                    old_file_path = new_file_path
                    print(f"Renamed {file} to {new_file}")
                elif file.endswith('.xkv'):
                    new_file = file[:-4]
                    new_file_path = os.path.join(root, new_file)
                    os.rename(old_file_path, new_file_path)
                    old_file_path = new_file_path
                    print(f"Renamed {file} to {new_file}")
                elif file.endswith('.jpg'):
                    new_file = os.path.splitext(file)[0] + '.mp4'
                    new_file_path = os.path.join(root, new_file)
                    os.rename(old_file_path, new_file_path)
                    old_file_path = new_file_path
                    print(f"Renamed {file} to {new_file}")
                elif file.endswith('.rmv1'):
                    new_file = os.path.splitext(file)[0] + '.rmvb'
                    new_file_path = os.path.join(root, new_file)
                    os.rename(old_file_path, new_file_path)
                    old_file_path = new_file_path
                    print(f"Renamed {file} to {new_file}")
                elif file.endswith('.mkv2'):
                    new_file = os.path.splitext(file)[0] + '.rmvb'
                    new_file_path = os.path.join(root, new_file)
                    os.rename(old_file_path, new_file_path)
                    old_file_path = new_file_path
                    print(f"Renamed {file} to {new_file}")
                elif file.endswith('.mkv1'):
                    new_file = os.path.splitext(file)[0] + '.rmvb'
                    new_file_path = os.path.join(root, new_file)
                    os.rename(old_file_path, new_file_path)
                    old_file_path = new_file_path
                    print(f"Renamed {file} to {new_file}")

                if old_file_path.endswith(('.mp4', '.avi', '.mkv', '.rmvb', '.ts', '.vob', 'm2ts', 
                                           '.webm', '.flv', '.mov', '.wmv', '.mpg', '.mpeg', '.m4v')):
                    unique_id = str(uuid.uuid4())
                    suffix = os.path.splitext(old_file_path)[1]
                    new_name_with_old_suffix = unique_id + suffix
                    rename_path = os.path.join(root, new_name_with_old_suffix)
                    os.rename(old_file_path, rename_path)  # 更改文件名，不改后缀
                    print(f"Renamed {old_file_path} to {rename_path}")

                    output_path = output_path or root # 如果没有指定output_path，则使用当前文件夹

                    final_path = os.path.join(output_path, f'{unique_id}.mp4')  # 最终存储路径
                    futures.append(executor.submit(process_movie, rename_path, final_path))
                else:
                    # 删除不是视频文件的文件
                    os.remove(old_file_path)
                    print(f"Deleted {old_file_path}")

        movie_count = len(futures)
        total_duration = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                continue
            movie_path, duration = result
            if duration is None:
                error_list.append(movie_path)
            else:
                total_duration += duration

        total_duration_str = str(timedelta(seconds=int(total_duration)))
        info = {
            'movie_count': movie_count,
            'total_duration': total_duration_str,
            'errors': error_list
        }
        json_name = info_json_name + "_info.json"
        with open(json_name, 'w') as f:
            json.dump(info, f)

    total_time_used_str = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Total process time: {total_time_used_str}")

# 注意在执行代码时，不要打开这两个文件夹及子文件夹，否则可能会出现访问权限错误
# process_folder(r'G:\test_movie\1080p_test', r'G:\test_movie\target')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='the path that save the videos', required=True)
    parser.add_argument('-o', '--output', help='the output path', required=False, default=None)
    parser.add_argument('-n', '--info_json_name', help="the name of info json", default="default")
    args = parser.parse_args()

    process_folder(args.path, args.output, args.info_json_name)
