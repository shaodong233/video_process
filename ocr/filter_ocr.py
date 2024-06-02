
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch

import easyocr
from PIL import Image

from argparse import ArgumentParser

def filter_ocr(ocr_model, video_path, frame_step=5, confidance_threshold=0.5, resolution_threshold=0.1, bbox_num_threshold=3, text_frame_num_threshold=0.2):
    clip = VideoFileClip(video_path)
    print("Processing video: ", video_path)
    frame_index = 0
    has_text_frame_num = 0

    text_frame_num_threshold = int(clip.fps * clip.duration * text_frame_num_threshold) // frame_step

    for frame in clip.iter_frames():
        if frame_index % frame_step == 0:  # Only process every frame_step frames
            try:
                image = frame[:, :, [2, 1, 0]]  # Convert RGB to BGR
                width, height = image.shape[1], image.shape[0]

                with torch.no_grad():
                    result = ocr_model.readtext(image)
                
                bbox_num = 0 
                region = 0

                for (bbox, text, conf) in result:
                    if conf > confidance_threshold:
                        point1, point2 = bbox[0], bbox[2]
                        text_width, text_height = point2[0] - point1[0], point2[1] - point1[1]
                        region += text_width * text_height
                        bbox_num += 1

                if bbox_num >= bbox_num_threshold or region >= width * height * resolution_threshold:
                    has_text_frame_num += 1
            
                if has_text_frame_num > text_frame_num_threshold:
                    break
            except Exception as e:
                return True
        frame_index += 1

    clip.close()
    return has_text_frame_num > text_frame_num_threshold

def filter_videos(txt_path, output_path, frame_step):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    
    with open(txt_path, 'r') as f:
        video_paths = f.read().splitlines()
    video_paths = sorted(video_paths)

    path_kv_dict = {video_path: os.path.join(output_path, os.path.basename(video_path)) for video_path in video_paths}
    video_paths = [video_path for video_path in video_paths if not os.path.exists(path_kv_dict[video_path])]
    video_paths = sorted(video_paths)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(filter_ocr, reader, path, frame_step): path for path in video_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing Videos'):
            video_path = futures[future]
            video_file = os.path.basename(video_path)
            try:
                has_text = future.result()
                if not has_text:
                    shutil.copy(video_path, os.path.join(output_path, video_file))
            except Exception as exc:
                print(f'Video file {video_file} generated an exception: {exc}')



if __name__ == "__main__":
    frame_step = 3  # Skip every 3 frames
    parser = ArgumentParser()
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output_ocr")
    args = parser.parse_args()
    txt_path = args.txt_path
    output_path = args.output_path
    filter_videos(txt_path, output_path, frame_step)
