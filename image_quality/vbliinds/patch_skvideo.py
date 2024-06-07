import os
import importlib.util

def patch_skvideo():
    # 获取 scikit-video 的安装路径
    spec = importlib.util.find_spec('skvideo')
    if spec is None:
        print("scikit-video is not installed")
        return

    skvideo_path = spec.submodule_search_locations[0]
    files_to_patch = [
        os.path.join(skvideo_path, 'io', 'abstract.py'),
        os.path.join(skvideo_path, 'measure', 'niqe.py'),
        './block.py',  # 修改路径到当前目录
        './vbliinds_frame_numba.py',
    ]

    for file_path in files_to_patch:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                import_added = False
                for line in lines:

                    line = line.replace('np.float', 'float')
                    line = line.replace('np.int', 'int')
                    line = line.replace('np.bool', 'bool')
                    line = line.replace('int8', 'np.int8')
                    line = line.replace('float32', 'np.float32')
                    line = line.replace('float64', 'np.float64')
                    line = line.replace('int32', 'np.int32')
                    line = line.replace('int64', 'np.int64')
                    line = line.replace('int16', 'np.int16')
                    file.write(line)

            print(f"Patched {file_path}")
        else:
            print(f"File {file_path} does not exist")

if __name__ == "__main__":
    patch_skvideo()
