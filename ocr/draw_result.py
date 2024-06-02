'''
根据ocr结果画出对应的图
'''

import argparse
import os
import json
from PIL import Image, ImageDraw
from paddleocr import draw_ocr
from decord import VideoReader, cpu

# 函数：读取 JSON 文件
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# 函数：从视频中抽取指定帧并绘制 OCR 结果
def draw_ocr_result(video_path, data, output_dir):
    # 打开视频文件
    vr = VideoReader(video_path, ctx=cpu(0))
    
    # 获取视频文件名，不带扩展名
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建输出目录
    output_subdir = os.path.join(output_dir, video_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    
    for result in data:
        frame_index = result['frame_index']
        frame = vr[frame_index]
        image = Image.fromarray(frame.asnumpy())

        # 获取 OCR 结果
        ocr_result = result['ocr_result']
        if not ocr_result:  # 检查是否有 OCR 结果
            continue

        for res in ocr_result:
            if not res:  # 检查 res 是否为空
                continue
            for line in res:
                if not line:  # 检查 line 是否为空
                    continue
                boxes = line[0]
                if not boxes:  # 检查 boxes 是否为空
                    continue
                txt = line[1][0]
                score = line[1][1]
                
                # 提取矩形的左上角和右下角
                min_x = min(box[0] for box in boxes)
                min_y = min(box[1] for box in boxes)
                max_x = max(box[0] for box in boxes)
                max_y = max(box[1] for box in boxes)
                rect = [(min_x, min_y), (max_x, max_y)]
                
                # 创建画布
                draw = ImageDraw.Draw(image)
                
                # 绘制 OCR 结果
                draw.rectangle(rect, outline='red', width=2)
                draw.text((min_x, min_y - 20), f'{txt} {score:.2f}', fill='red', font=None)
        
        # 保存结果图像
        result_image_path = os.path.join(output_subdir, f'{video_name}_frame_{frame_index}.jpg')
        image.save(result_image_path)

# 函数：处理指定目录下的所有 JSON 文件
def process_json_files(input_dir, output_dir):
    # 遍历输入目录中的所有 JSON 文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            # 加载 JSON 文件
            json_path = os.path.join(input_dir, filename)
            data = load_json(json_path)
            
            # 获取视频路径
            video_path = os.path.splitext(json_path)[0] + '.mp4'

            # 绘制 OCR 结果
            draw_ocr_result(video_path, data, output_dir)

# 示例用法
process_json_files("/home/yeyang/movie_test/ocr_test", "/home/yeyang/movie_test/ocr_test/output")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-p', "--path", help="the path that save the videos", required=True)
#     parser.add_argument('-o', "--output", help="the path that save the output videos", required=True)
    
#     args = parser.parse_args()
#     process_json_files(args.path, args.output)
