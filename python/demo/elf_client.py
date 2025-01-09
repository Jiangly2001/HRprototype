from config import Config
from dataset import VideoLoader
from flow import FlowControl
from inference_model import InferenceModelDetectron2
from util.helper import display_imgs
from util.myutil import get_file_list

run_ground_truth_mode = False

if __name__ == "__main__":
    config = Config()

    if config.visualization_mode:
        import cv2

    # video_dataset = VideoLoader(
    #     video_dir_root=config.video_dataset_dir,
    # )
    files = get_file_list(config.video_dataset_dir, extensions=[".jpg", ".png"])

    model = InferenceModelDetectron2()
    if run_ground_truth_mode:
        model.create_model()
        print("Model created.")

    with FlowControl(config, model) as flow_control:
        for i, file in enumerate(files):
            img = cv2.imread(file)
            inference_result = flow_control.run(img)
            render_img = model.render(img, inference_result)
            if run_ground_truth_mode:
                gt_inference_result = model.run(img)
                gt_render_img = model.render(img, gt_inference_result)

            if config.visualization_mode:
                display_imgs(render_img, "Our Approach", i)
                if run_ground_truth_mode:
                    display_imgs(gt_render_img, "Full frame (non-Elf) offloading")
                cv2.waitKey(5)
