from typing import Any, Optional, Union, Tuple, List, Dict
import os
import sys
import time
import numpy as np
import abc
import cv2
import collections
# Add current path
from ..preprocess import torch_proc, torch_processor, multiproc_torch
from ..utils import euclidean


def load_frames(siamese: Any, golden_paths: List[str], sample_paths: List[str]) -> Dict[str, List[np.ndarray]]:
    ret: Dict[str, List[np.ndarray]] \
        = collections.defaultdict(list)

    input_names = [name for name in siamese.input_shapes.keys()]
    input_paths = [golden_paths, sample_paths]
    for name, paths in zip(input_names, input_paths):
        for path in paths:
            if not os.path.isfile(path):
                continue
            ret[name].append(cv2.imread(path))
    if ret is None:
        raise RuntimeError('Something went wrong.')
    return ret


class BasicSiameseIO:
    """Support the inputs and outputs of Siamese Model
    """

    input_shapes: Dict[str, list] = {"input_1": [
        1, 3, 224, 224], "input_2": [1, 3, 224, 224]}
    output_shapes: Dict[str, list] = {"output_1": [1, 5], "output_2": [1, 5]}

    def __init__(self, input_shapes: Dict[str, list], output_shapes: Dict[str, list]) -> None:
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

    def preprocess(self, inputs: Dict[str, List[np.ndarray]]):
        assert len(
            inputs.keys()) == 2, f"The Siamese Model should has two inputs. support shape is {self.input_shapes}, but got {len(inputs.keys())} "
        blob = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self.input_shapes.items()
        }
        for name, frames in inputs.items():
            data = multiproc_torch(frames, self.input_shapes[name][2:])
            blob[name][0:data.shape[0], :, :, :] = data
        return blob

    def postprocess(self, outputs: List[np.ndarray]) -> List[float]:
        for idx, (_, shape) in enumerate(self.output_shapes.items()):
            if tuple(shape) == outputs[idx].shape:
                continue
            outputs[idx] = outputs[idx].reshape(shape)
        return euclidean((outputs[0], outputs[1]))


if __name__ == "__main__":
    # sia = Siamese(input_shapes=[16, 3, 256, 256])

    input_shapes = {
        "input_1": [16, 3, 256, 256],
        "input_2": [16, 3, 256, 256]
    }
    output_shapes = {
        "output_1": [1, 5],
        "output_2": [1, 5]
    }

    basic = BasicSiameseIO(
        input_shapes=input_shapes,
        output_shapes=output_shapes
    )

    frame = np.ones([256, 256, 3])
    num_of_frame = 10
    inputs = {
        "input_1": [frame for _ in range(num_of_frame)],
        "input_2": [frame for _ in range(num_of_frame)]
    }
    basic.preprocess(inputs)
