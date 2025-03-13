from config import Config
from flow import ProcessingFlowControl
from inference_model import InferenceModelDetectron2
from inference_model import ViTInferenceDetectron2

from util.helper import display_imgs
from util.myutil import get_file_list


if __name__ == "__main__":
    config = Config()

    if config.visualization_mode:
        import cv2

    files = get_file_list(config.video_dataset_dir, extensions=[".jpg", ".png"])

    model = ViTInferenceDetectron2()
    model.create_model()
    print("Model created.")

    with ProcessingFlowControl(config, model) as processing_flow_control:
        for i, file in enumerate(files):
            img = cv2.imread(file)
            inference_result = processing_flow_control.run(img)
            render_img = model.render(img, inference_result)

            if config.visualization_mode:
                display_imgs(render_img, "Our Approach", i)
                cv2.waitKey(5)
