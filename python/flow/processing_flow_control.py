import os
import random
from typing import List, Any, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans
from canvas_packing import GrowingPacker
from canvas_packing.util import sort_key
from config import Config
from flow.flow_data import FlowData
from flow.offload import (
    send_frame,
    receive_inference_results,
    get_random_socket,
)
from inference_model import InferenceModelTest, InferenceModelInterface
from inference_model import ViTInferenceDetectron2
from networking import EncoderPickle, EncoderBase, connect_sockets, close_sockets
from rp_partition import RPPartitioner
from rp_predict import RPPredictor
from rp_predict.util import render_bbox
from inference_model import ViTInferenceDetectron2

from util.helper import display_imgs
from util.myutil import tensor_to_list, predict_encoding_size

class ProcessingFlowControl:
    """This class controls how ours approach processes each video frame."""

    def __init__(
        self,
        config: Config = Config(),
        inference_model: InferenceModelInterface = InferenceModelTest()
    ):
        self._config: Config = config
        self._rp_predictor: RPPredictor = RPPredictor(config)
        self._rp_partitioner: RPPartitioner = RPPartitioner(config)
        self._flow_data: FlowData = FlowData()
        self._flow_data.create_executor()

        """A machine learning model to run inference."""
        self.inference_model = inference_model

        """An encoder to encode&decode video frames."""
        self._frame_encoder: EncoderBase = EncoderPickle()

        """If the flag is set True, it shows the received frame and frame partitions."""
        self.visualization_mode: bool = config.visualization_mode

    def run(
        self,
        frame: np.ndarray
    ) -> Any:
        """
        It takes a video frame as the input and returns the model inference result.
        :param frame: np.ndarray, an input video frame.
        :return: model inference result.
        """
        self._register_new_frames(frame)
        bd = self._bandwidth_predictor()
        # print(f"Predicted Bandwidth is {bd} Mbps.")
        # if bd is None:
        #     self._run_cold_start(frame)
        #     return self._flow_data.inference_results[0]

        # -------------Edge Device ViT Inference-------------
        predictions, attention_weights = self.inference_model.run(frame)
        self._flow_data.inference_results.append(predictions)
        # -------------PROCESSING BEFORE Transmission----------------
        # Step 1: Load bounding box and token importance
        instances = predictions['instances']
        bboxes = tensor_to_list(instances.pred_boxes.tensor)
        scores = tensor_to_list(instances.scores)

        global_importance = 0
        for idx, attn in enumerate(attention_weights):
            if idx in [2, 5, 8, 11]:
                layer_importance = attn.sum(dim=-2).mean(dim=-2)
                global_importance += layer_importance
        global_importance /= 4
        token_importance = global_importance[:2304].detach().cpu()
        token_importance = (token_importance - token_importance.min()) / (
                token_importance.max() - token_importance.min())

        # Step 2: Cluster token importance into CLASS 0, 1, 2
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(token_importance.reshape(-1, 1))
        # Compute the mean token importance for each cluster
        cluster_means = {i: np.mean(token_importance[clusters == i].numpy()) for i in range(3)}
        # Sort clusters by their mean importance values (smallest to largest)
        sorted_clusters = sorted(cluster_means, key=cluster_means.get)
        # Reassign cluster labels: smallest = 0, middle = 1, largest = 2
        cluster_mapping = {sorted_clusters[i]: i for i in range(3)}
        sorted_cluster_labels = np.vectorize(cluster_mapping.get)(clusters)
        clusters = sorted_cluster_labels

        # Step 3: Define partition parameters
        image_height, image_width = self._config.frame_height, self._config.frame_width
        partition_height, partition_width = 1020, 1020
        # partition_height, partition_width = 960, 960
        patch_size = image_width * 16 // 1024
        patch_num_row, patch_num_col = int(image_height // patch_size), int(image_width // patch_size)
        partition_param = clusters.reshape(patch_num_row, patch_num_col)
        # mark patches with bbox overlap
        bbox_patches = np.zeros_like(partition_param, dtype=bool)
        for idx in range(len(bboxes)):
            if scores[idx] > 0.93:
                x1, y1, x2, y2 = bboxes[idx]
                row_start = max(0, int(y1 // patch_size))
                row_end = min(patch_num_row, int(np.ceil(y2 / patch_size)))
                col_start = max(0, int(x1 // patch_size))
                col_end = min(patch_num_col, int(np.ceil(x2 / patch_size)))
                bbox_patches[row_start:row_end, col_start:col_end] = True  # 将这些 patch 标记为与 bbox 有交集
                # partition_param[row_start:row_end, col_start:col_end] = (partition_param[row_start:row_end, col_start:col_end] * 0.5).astype(int)
        overlap_height_ratio, overlap_width_ratio = 0.25, 0.25
        step_height = int(partition_height * (1 - overlap_height_ratio))
        step_width = int(partition_width * (1 - overlap_width_ratio))

        # Step 4: Iterate over image partitions
        transfer_config = np.full_like(partition_param, -2, dtype=int)
        for row in range(0, image_height + step_height - partition_height, step_height):
            for col in range(0, image_width + step_width - partition_width, step_width):
                row_end = row + partition_height
                col_end = col + partition_width
                if row_end > image_height:
                    row_end = image_height
                    row = image_height - partition_height
                    row_edge = True
                if col_end > image_width:
                    col_end = image_width
                    col = image_width - partition_width
                    col_edge = True
                # Get the tokens clusters in this patch (row,col) is a "Z" seq
                row_start_idx = row // patch_size
                row_end_idx = row_end // patch_size
                col_start_idx = col // patch_size
                col_end_idx = col_end // patch_size
                partition_patches = partition_param[row_start_idx:row_end_idx, col_start_idx:col_end_idx]
                partition_bbox_patches = bbox_patches[row_start_idx:row_end_idx, col_start_idx:col_end_idx]
                class2_patches = (partition_patches == 2)
                # Step 4.1: If all tokens in the patch belong to CLASS 0 or 1, skip
                # Step 4.2: Check intersection of CLASS 2 tokens with bbox, skip if there's overlap
                if np.all(partition_patches < 2) or np.all(
                        partition_bbox_patches[class2_patches]):  # do not transmit current partition
                    transfer_config_slice = transfer_config[row_start_idx:row_end_idx, col_start_idx:col_end_idx]
                    transfer_config[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = np.where(
                        transfer_config_slice == -2, -1, transfer_config_slice)
                else:  # transmit current partition
                    transfer_config[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = partition_param[
                                                                                            row_start_idx:row_end_idx,
                                                                                            col_start_idx:col_end_idx]

        # Step 5: Decide Transmit Configeration per Patch
        if self._config.fixed_qp_for_class is None:
            n_patches = {k + 1: np.bincount(transfer_config.flatten(), minlength=3)[k] for k in range(3)}
            best_combination, final_latency, meet_latency = self._qp_scheduler(bd, n_patches)
            print(f"QP combination: {best_combination}, Predicted Latency: {final_latency}, Meet Latency: {meet_latency}")
        else:
            best_combination = self._config.fixed_qp_for_class
            print(f"User set a fixed QP: {best_combination}")
        # Step 6: Generate ROI Parameters
        transfer_config = np.vectorize(lambda i: best_combination[i])(transfer_config)
        blocks = []
        for i in range(patch_num_col):
            for j in range(patch_num_row):
                y_start = j * patch_size
                x_start = i * patch_size
                patch = frame[x_start:x_start + patch_size, y_start:y_start + patch_size]
                rate = transfer_config[i, j]
                if rate == -1:
                    continue
                if rate > 1:
                    downsampled_patch = cv2.resize(patch, (patch_size // rate, patch_size // rate),
                                                   interpolation=cv2.INTER_AREA)
                else:
                    downsampled_patch = patch

                avg_color = np.mean(downsampled_patch, axis=(0, 1))
                blocks.append({"image": downsampled_patch, "w": patch_size // rate, "h": patch_size // rate,
                               "color": avg_color})

        # Sort patches by color consistency
        blocks.sort(key=sort_key, reverse=True)

        packer = GrowingPacker()
        packer.fit(blocks)
        # Create packed image
        max_width, max_height = packer.root["w"], packer.root["h"]
        packed_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        for block in blocks:
            if "fit" in block and block["fit"]:
                x, y = block["fit"]["x"], block["fit"]["y"]
                packed_image[y:y + block["h"], x:x + block["w"]] = block["image"]

        if self.visualization_mode:
            display_imgs(packed_image,"Ours internal process (Transmitted Image)", self._flow_data.frame_count)

        # Sending Original High Resolution Frame
        # self._run_warm_up(frame, roi_param)
        self._run_cold_start(frame)

        device_predictions = self._flow_data.inference_results[0]
        cloud_predictions = self._flow_data.inference_results[1]
        return self._weighted_ensembling(device_predictions,cloud_predictions)

    def __enter__(self):
        self.connect_remote_servers()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def connect_remote_servers(self) -> None:
        """Connect remote servers that are available for model inference and LRC service."""
        self._flow_data.sockets = connect_sockets(
            self._config.servers
        )

        # TODO: Dynamically assign LRC server.
        self._flow_data._lrc_socket = connect_sockets(
            [random.choice(self._config.servers)]
        )[0]

    def shutdown(self) -> None:
        self._flow_data.executor.shutdown()
        close_sockets(
            self._flow_data.sockets
        )

    def _register_new_frames(self, frame: np.ndarray) -> None:
        """Update Elf status when receiving a new frame."""
        self._flow_data.cur_frame = frame
        self._flow_data.frame_count += 1
        self._flow_data.offloading_tasks = []
        self._flow_data.inference_results = []
        self._config.frame_height, self._config.frame_width = frame.shape[:2]
        print(f"Process frame index {self._flow_data.frame_count}.")

    def _run_cold_start(
        self,
        frame: np.ndarray
    ) -> None:
        socket = get_random_socket(
            self._flow_data.sockets
        )

        # Sending Original High Resolution Frame
        send_frame(
            socket,
            frame,
            self._frame_encoder,
        )

        # Receiving Inference Results
        inference_result = receive_inference_results(socket)
        self._flow_data.inference_results.append(inference_result)


    def _run_warm_up(
        self,
        frame: np.ndarray,
        roi_param: np.ndarray,
    ) -> None:
        socket = get_random_socket(
            self._flow_data.sockets
        )

        # Sending Original High Resolution Frame
        send_frame(
            socket,
            frame,
            self._frame_encoder,
            roi_param=roi_param,
        )

        # Receiving Inference Results
        inference_result = receive_inference_results(socket)
        self._flow_data.inference_results.append(inference_result)


    def _qp_scheduler(self, predicted_bandwidth):
        qp = self._config.qp_candidates
        classes = self._config.classes
        latency_constraint = self._config.latency_constraint
        best_combination = None
        # Initialize rates for all classes to the smallest r
        current_rates = {c: qp[0] for c in classes}
        # Calculate total latency for the current combination
        cur_latency = self._calculate_latency(current_rates, predicted_bandwidth)
        # If latency constraint is satisfied, record the combination
        if cur_latency <= latency_constraint:
            best_combination = current_rates.copy()
            final_latency = cur_latency
            return best_combination, final_latency, True

        # Find the next combination by increasing the rate of the lowest-priority class
        for c in classes:
            for r in qp[1:]:
                current_rates[c] = r
                cur_latency = self._calculate_latency(current_rates, predicted_bandwidth)
                if cur_latency <= latency_constraint:
                    best_combination = current_rates.copy()
                    final_latency = cur_latency
                    return best_combination, final_latency, True

        best_combination = current_rates.copy()
        final_latency = cur_latency
        return best_combination, final_latency, False

    def _calculate_latency(self, rates, bandwidth, n_patches):
        """Calculate the total latency for the current combination."""
        predicted_size = predict_encoding_size(rates, n_patches, self._flow_data.last_encoding_size)
        return predicted_size / bandwidth

    def _merge_partitions(
        self,
        offsets: List[int],
    ) -> Any:
        """ Merge the model inference results from different servers."""
        return self.inference_model.merge(
            self._flow_data.inference_results,
            offsets,
            frame_height=self._config.frame_height,
            frame_width=self._config.frame_width,
            merge_mask=self._config.merge_mask,
        )

    def _bandwidth_predictor(self) -> Optional[float]:
        if len(self._flow_data.bandwidth_records) < 1:
            self._flow_data.bandwidth_records.append(50) # TODO: change to real bandwidth assessment
            return None

        return np.mean([1 / x for x in self._flow_data.bandwidth_records[-3:]])

    def _weighted_ensembling(self, device_predictions, cloud_predictions):
        """
        Combines the predictions from the edge and cloud models using weighted box fusion.

        Args:
            device_predictions (np.ndarray): The predictions from the edge model.
            cloud_predictions (np.ndarray): The predictions from the cloud model.

        Returns:
            np.ndarray: The combined predictions.
        """
        # Calculate the weights for the edge and cloud models
        edge_weight = self._config.edge_model_weight
        cloud_weight = self._config.cloud_model_weight

        # Calculate the weighted average of the bounding box coordinates
        x1 = edge_weight * device_predictions[:, 0] + cloud_weight * cloud_predictions[:, 0]
        y1 = edge_weight * device_predictions[:, 1] + cloud_weight * cloud_predictions[:, 1]
        x2 = edge_weight * device_predictions[:, 2] + cloud_weight * cloud_predictions[:, 2]
        y2 = edge_weight * device_predictions[:, 3] + cloud_weight * cloud_predictions[:, 3]

        # Calculate the weighted average of the class probabilities
        class_probabilities = edge_weight * device_predictions[:, 5:] + cloud_weight * cloud_predictions[:, 5:]

        # Combine the results
        combined_predictions = np.concatenate((x1[:, None], y1[:, None], x2[:, None], y2[:, None], class_probabilities),
                                              axis=1)

        return combined_predictions

