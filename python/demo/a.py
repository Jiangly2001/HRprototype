import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 读取 Token 重要性数据
csv_path = r"F:\B\result\ours\token_importance.csv"
df = pd.read_csv(csv_path, header=None)

# 解析 token_importance.csv，按 scene_id (SEQ_XX) 进行分组求和
scene_importance = {}

df['scene_prefix'] = df[0].apply(lambda x: "_".join(x.split('_')[:2]))  # 提取 SEQ_XX
df_values = df.iloc[:, 1:]  # 取数值部分（去掉文件名列）
df_selected = df_values[:2304]  # 取前 2304 列
df_values = df_selected.apply(pd.to_numeric, errors='coerce')  # 将所有数据转换为数值，无法转换的设为 NaN

# 2. 按 scene_prefix 进行分组，并计算每列之和
scene_importance = df_values.groupby(df['scene_prefix']).sum()

# 输出
# print(scene_importance)



def get_file_list(directory, extensions=None):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions:
                if filename.endswith(tuple(extensions)):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
    return sorted(files)


# 2. 归一化处理（避免除零错误）
importance_values = scene_importance.to_numpy()  # 获取 NumPy 数组

importance_min = np.min(importance_values)
importance_max = np.max(importance_values)

# If you need a list of values
importance_values_list = importance_values.tolist()

if importance_max > importance_min:
    patch_importance = [(v - importance_min) / (importance_max - importance_min) for v in importance_values]
else:
    patch_importance = [0 for _ in importance_values]  # 处理所有值相同的情况

# 3. 加载图像
video_dataset_dir = r"F:\Datasets\gigapixel\video\train\panda_video_4k"
for scene_id, scene_dir in enumerate(os.listdir(video_dataset_dir)):
    scene_path = os.path.join(video_dataset_dir, scene_dir)
    files = get_file_list(scene_path, extensions=[".jpg", ".png"])

    if not files:  # Skip empty directories
        continue

    image_path = files[0]
    image = Image.open(image_path)

    # Adjust image size
    max_dim = 1024
    ratio = max_dim / max(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size)

    new_image = Image.new("RGB", (max_dim, max_dim))
    new_image.paste(image, (0, 0))  # Align to the top

    image_array = np.array(new_image)
    patch_size = max_dim // 64  # Assuming 64 patches per dimension

    image_patches = [image_array[i:i + patch_size, j:j + patch_size] for i in range(0, max_dim, patch_size) for j in
                     range(0, max_dim, patch_size)]


    # 4. 可视化
    def get_color(value):
        """Generate a color based on importance value."""
        return (value, 0, 0, value)  # Red channel with transparency


    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_array)

    # 确保 64x64 = 4096 个块
    for idx, patch in enumerate(image_patches):
        row = (idx // 64) * patch_size
        col = (idx % 64) * patch_size
        color = get_color(patch_importance[scene_id][idx].item())
        rect = patches.Rectangle((col, row), patch_size, patch_size, linewidth=1, edgecolor='none', facecolor=color)
        ax.add_patch(rect)

    plt.axis('off')
    plt.title(f"Scene: {scene_id + 1}, Importance: {patch_importance[scene_id]:.2f}")
    plt.show()
