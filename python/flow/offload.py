import concurrent.futures
import csv
import random
import zlib
from typing import List, Any, Optional

import gevent
import numpy as np
from zmq.sugar.socket import Socket

from networking import EncoderBase
from PIL import Image
import io


def send_frame(
    socket: Socket,
    frame: np.ndarray,
    frame_encoder: EncoderBase,
    file_path: str,
    roi_param: Optional[Any] = None,
    csv_file_path: Optional[str] = r"F:\B\result\elf\encoded_frame_size.csv"
) -> None:
    """Encode a frame and send it to a remote socket."""
    if roi_param is None:
        encoded_frame = frame_encoder.encode(frame)
    else:
        # Perform RoI encoding using roi_param
        encoded_frame = frame_encoder.encode(frame, roi_param)
    # print(len(encoded_frame))
    # compressed_data = zlib.compress(encoded_frame)  # 压缩后的数据
    # print(len(compressed_data))
    # quality = 75
    # img = Image.fromarray(frame)
    # img_bytes = io.BytesIO()
    # img.save(img_bytes, format="JPEG", quality=quality)  # 设置 JPEG 质量
    # compressed_data_jpeg = img_bytes.getvalue()
    # print(len(compressed_data_jpeg))
    quality = 100
    img = Image.fromarray(frame)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG", quality=quality)  # 设置 JPEG 质量
    compressed_data_jpeg = img_bytes.getvalue()
    # print(len(compressed_data_jpeg))
    # print(len(encoded_frame)/len(compressed_data_jpeg))
    socket.send(encoded_frame, copy=False)

    # 将编码帧大小和文件路径保存到 CSV 文件
    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([file_path, len(compressed_data_jpeg)])
    return None


def submit_offloading_tasks(
    frames: List[np.ndarray],
    frame_encoder: EncoderBase,
    sockets: List[Socket],
    executor: concurrent.futures.ThreadPoolExecutor,
    file_path: str,
) -> List[concurrent.futures.Future]:
    """Submit offloading tasks to the connected sockets."""
    offloading_tasks = [
        executor.submit(
            send_frame,
            socket,
            frame,
            frame_encoder,
            file_path,
        ) for frame, socket in zip(frames, sockets)
    ]

    return offloading_tasks


def wait_offload_finished(
    offloading_tasks: List[concurrent.futures.Future]
) -> None:
    """Wait offloading to be done."""

    concurrent.futures.wait(offloading_tasks)


def receive_inference_results(
    socket: Socket,
) -> Any:
    """Save the result returned by a remote socket to a list."""
    return socket.recv_pyobj()


def schedule_inference_results_collection(
    sockets: List[Socket],
) -> List[Any]:
    """Schedule to collect inference results from remote servers."""

    tasks = [
        gevent.spawn(
            receive_inference_results,
            sockets[i],
        ) for i in range(len(sockets))
    ]

    return tasks


def collect_inference_results(
    sockets: List[Socket],
    lrc_socket: Optional[Socket] = None,
) -> List[Any]:
    """Collect inference results from remote servers."""

    inference_results_collection_tasks = schedule_inference_results_collection(
        sockets,
    )

    if lrc_socket is not None:
        inference_results_collection_tasks += schedule_inference_results_collection(
            [lrc_socket],
        )

    gevent.joinall(
        inference_results_collection_tasks
    )

    return [task.value for task in inference_results_collection_tasks]


def get_random_socket(sockets: List[Socket]) -> Socket:
    return random.choice(sockets)
