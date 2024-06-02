# 记录初始文件夹结构并保存为JSON文件
# python record.py -p /path/to/directory -o /path/to/output.json

import os
import json
import argparse

def get_directory_structure(root_dir):
    """
    Creates a nested dictionary that represents the folder structure of root_dir.
    """
    directory_structure = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Create a nested dictionary for the current directory
        path_dict = directory_structure
        parts = os.path.relpath(dirpath, root_dir).split(os.sep)
        for part in parts:
            path_dict = path_dict.setdefault(part, {})
        
        # Add files to the current directory
        path_dict.update({file: None for file in filenames})

    return directory_structure

def save_to_json(data, json_path):
    """
    Saves the provided data to a JSON file.
    """
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Record directory structure and save to JSON.")
    parser.add_argument('-p', '--path', required=True, help="Path to the directory to record")
    parser.add_argument('-o', '--output', required=True, help="Output JSON file path")
    args = parser.parse_args()

    root_dir = args.path
    json_path = args.output

    # Get directory structure
    directory_structure = get_directory_structure(root_dir)

    # Save to JSON
    save_to_json(directory_structure, json_path)

    print(f"Directory structure saved to {json_path}")

if __name__ == "__main__":
    main()