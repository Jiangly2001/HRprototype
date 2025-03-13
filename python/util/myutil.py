import os
import torch
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_file_list(directory, extensions=None):
    """
    Given a directory, return a list of its files, sorted alphabetically.

    Parameters
    ----------
    directory : str
        The path to the directory.
    extensions : tuple of str, optional
        Only include files with these extensions.

    Returns
    -------
    list of str
        The paths of the files.

    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions:
                if filename.endswith(tuple(extensions)):
                    files.append(os.path.join(root, filename))
            else:
                files.append(os.path.join(root, filename))
    return sorted(files)

def tensor_to_list(tensor):
    return tensor.cpu().detach().numpy().tolist() if isinstance(tensor, torch.Tensor) else tensor


def predict_encoding_size(rates, n_patches, last_encoding_size):
    """
    Predict the size of a tensor after encoding, in bytes.

    Parameters
    ----------
    rates : float
        The compression rate.
    n_patches : int
        The number of patches in the tensor.
    last_encoding_size : int
        The size of the tensor after previous encoding.

    Returns
    -------
    int
        The predicted size of the tensor in bytes.

    """
    return int(last_encoding_size * rates * n_patches)





def save_all_inference_results(inference_results, image_paths, output_path):
    # 构建符合 COCO 格式的数据结构
    result_dict = {
        "images": [
            {
                "id": frame_id,
                "file_name": os.path.basename(image_path),  # 图像文件名
                "width": 3840,  # 图像宽度
                "height": 2160,  # 图像高度
            }
            for frame_id, image_path in enumerate(image_paths)
        ],
        "annotations": [
            {
                "image_id": frame_id,
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # [x_min, y_min, width, height]
                "category_id": int(class_id),
                "score": float(score),
            }
            for frame_id, result in inference_results.items()
            for box, score, class_id in zip(result["boxes"], result["scores"], result["classes"])
        ]
    }

    # 保存为 JSON 文件
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=4)


# 对比分析并计算准确率
def evaluate_accuracy(coco_annotations, inference_results):
    coco_results = []
    for frame_id, result in inference_results.items():
        for box, score, class_id in zip(result["boxes"], result["scores"], result["classes"]):
            coco_results.append({
                "image_id": frame_id,
                "category_id": int(class_id),
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # [x_min, y_min, width, height]
                "score": float(score),
            })
    coco_dt = coco_annotations.loadRes(coco_results)
    coco_eval = COCOeval(coco_annotations, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP@[0.5:0.95]