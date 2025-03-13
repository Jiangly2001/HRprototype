import os
import time

from config import Config
from dataset import VideoLoader
from flow import FlowControl
from inference_model import InferenceModelDetectron2
from util.helper import display_imgs
from util.myutil import get_file_list, save_all_inference_results, evaluate_accuracy
run_ground_truth_mode = False
import cv2

if __name__ == "__main__":
    config = Config()

    # video_dataset = VideoLoader(
    #     video_dir_root=config.video_dataset_dir,
    # )

    model = InferenceModelDetectron2()
    if run_ground_truth_mode:
        model.create_model()
        print("Model created.")

    for scene_id, scene_dir in enumerate(os.listdir(config.video_dataset_dir)):
        scene_path = os.path.join(config.video_dataset_dir, scene_dir)
        if os.path.isdir(scene_path):  # 确保是文件夹
            print(f"正在处理场景 {scene_id}: {scene_path}")
        files = get_file_list(scene_path, extensions=[".jpg", ".png"])

        inference_results = {}
        output_path = rf"F:\B\result\elf\Scene{scene_id}_inference_results_with_image_path.json"
        with FlowControl(config, model) as flow_control:
            for i, file in enumerate(files):
                img = cv2.imread(file)
                inference_result = flow_control.run(img, file)
                inference_results[i] = {
                    "boxes": inference_result["instances"].pred_boxes.tensor.tolist(),
                    "scores": inference_result["instances"].scores.tolist(),
                    "classes": inference_result["instances"].pred_classes.tolist(),
                }
                render_img = model.render(img, inference_result)
                if run_ground_truth_mode:
                    gt_inference_result = model.run(img)
                    gt_render_img = model.render(img, gt_inference_result)

                if config.visualization_mode:
                    display_imgs(render_img, "Our Approach", i)
                    if run_ground_truth_mode:
                        display_imgs(gt_render_img, "Full frame (non-Elf) offloading")
                    cv2.waitKey(5)

        # save_all_inference_results(inference_results, files, output_path)
        print(f"所有帧的检测结果和 image_path 已保存到 {output_path}")
