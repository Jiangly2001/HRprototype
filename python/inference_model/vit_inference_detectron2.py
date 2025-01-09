"""
This class is an example integration with Detectron2:
https://github.com/facebookresearch/detectron2.
"""
import sys
from pathlib import Path
from typing import List, Any

import numpy as np
import torch

from detectron2.data.transforms import ResizeShortestEdge
from inference_model import InferenceModelInterface

# sys.path.append(str(Path.home()) + '/detectron')
"""This is the path to install the source code of detectron."""

from detectron.demo.vit_setup import maskrcnn_interface, render
"""Please follow the instruction in REAMDE to add the API maskrcnn_interface under detectron/demo/demo.py."""


class ViTInferenceDetectron2(InferenceModelInterface):
    def create_model(self):
        self.app = maskrcnn_interface()

    def run(self, img: np.ndarray) -> Any:
        height, width = img.shape[:2]
        aug = ResizeShortestEdge(short_edge_length=1024, max_size=1024)
        image = aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # image.to('cuda')
        # with torch.no_grad():
        #     self.predictions, self.attention_weights = self.app(image)
        inputs = {"image": image, "height": height, "width": width}
        # TODO: move the tensors to the CUDA device
        with torch.no_grad():
            # TODO:self.predictions, self.attention_weights = self.app([inputs])
            self.predictions = self.app([inputs])
        return self.predictions[0]

    @staticmethod
    def extract_rps(inference_result: Any) -> np.ndarray:
        return inference_result['instances'].get_fields()['pred_boxes'].tensor.numpy()

    @staticmethod
    def extract_masks(inference_result: Any) -> torch.tensor:
        return inference_result['instances'].get_fields()['pred_masks']

    def render(self, img: np.ndarray, inference_result: Any) -> np.ndarray:
        return render(img, inference_result).get_image()

    def merge(
        self,
        inference_results: List[Any],
        offsets: List[List[int]],
        **kwargs
    ) -> Any:

        frame_height = kwargs.get("frame_height")
        frame_width = kwargs.get('frame_width')
        merge_mask = kwargs.get('merge_mask')

        for index, inference_result in enumerate(inference_results):
            if len(inference_result["instances"]) == 0:
                continue

            w, h = offsets[index]

            # Offset boxes.
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 0] += w
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 1] += h
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 2] += w
            inference_result["instances"].get_fields()["pred_boxes"].tensor[:, 3] += h

            # Offset mask.
            if merge_mask:
                shape = inference_result["instances"].image_size
                pad = torch.nn.ConstantPad2d((w, frame_width - w - shape[1],
                                              h, frame_height - h - shape[0]),
                                             0)
                inference_result["instances"].get_fields()["pred_masks"] = pad(
                    inference_result["instances"].get_fields()["pred_masks"][:, :, ])

        index = 0
        while index < len(inference_results) and len(inference_results[index]["instances"]) == 0:
            index += 1

        # Return if the current frame contains zero object of interest.
        if index == len(inference_results):
            return inference_results[0]

        ans = inference_results[index]

        ans["instances"]._image_size = (frame_height, frame_width)
        for i in range(index+1, len(inference_results)):
            if len(inference_results[i]["instances"]) == 0:
                continue

            ans["instances"].get_fields()["pred_boxes"].tensor = torch.cat(
                [ans["instances"].get_fields()["pred_boxes"].tensor,
                 inference_results[i]["instances"].get_fields()["pred_boxes"].tensor], dim=0)

            ans["instances"].get_fields()["scores"] = torch.cat(
                [ans["instances"].get_fields()["scores"],
                 inference_results[i]["instances"].get_fields()["scores"]], dim=0)

            ans["instances"].get_fields()["pred_classes"] = torch.cat(
                [ans["instances"].get_fields()["pred_classes"],
                 inference_results[i]["instances"].get_fields()["pred_classes"]], dim=0)

            if merge_mask:
                ans["instances"].get_fields()["pred_masks"] = torch.cat(
                    [
                        ans["instances"].get_fields()["pred_masks"],
                        inference_results[i]["instances"].get_fields()["pred_masks"],
                    ],
                    dim=0
                )

        return ans