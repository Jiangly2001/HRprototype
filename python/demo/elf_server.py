import sys
from pathlib import Path
import zmq
import zmq.asyncio
from inference_model import InferenceModelDetectron2
from inference_model import ViTInferenceDetectron2
from config import Config
from networking import EncoderPickle
sys.path.append(str(Path.home()) + '/detectron')


visualization_mode = False
if visualization_mode:
    import cv2
    from util.helper import display_imgs


def run(port: int):
    model = InferenceModelDetectron2()
    # model = ViTInferenceDetectron2()
    model.create_model()
    print("Model created.")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind(f"tcp://*:{port}")
    except zmq.error.ZMQError:
        print(f"Cannot establish the service at port {port}")

    print(f"Start to running service at port {port}")

    encoder = EncoderPickle()

    while True:
        arr = socket.recv()
        print("Frame received from edge device.")

        img = encoder.decode(arr)
        print(img.shape)

        predictions = model.run(img)
        if visualization_mode:
            render_img = model.render(
                img,
                predictions
            )
            display_imgs(render_img, "server processing")
            cv2.waitKey(5)

        predictions['instances'] = predictions['instances'].to('cpu')

        socket.send_pyobj(predictions, copy=False)
        print("Predictions sent back to client.")


if __name__ == "__main__":
    config = Config()
    run(config.servers[config.server_index][1])
