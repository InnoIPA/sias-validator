from typing import Any, List
import numpy as np
import cv2
from collections import namedtuple


class Offsetter:

    def __init__(self,
                 mean: list = [0.0, 0.0, 0.0],
                 std: list = [1.0, 1.0, 1.0],
                 channel: int = 3) -> None:
        self.mean: np.ndarray = np.array(mean)
        self.std: np.ndarray = np.array(std)
        self.channel: int = channel

    def __call__(self, input: np.ndarray) -> np.ndarray:
        h, w, c = input.shape
        assert c == self.channel, f"Support input channel is {self.channel}"
        for i in range(c):
            input[:, :, i] = (input[:, :, i]-self.mean[i])/self.std[i]
        return input


normalizer = Offsetter(std=[255.0, 255.0, 255.0])
torch_offsetter = Offsetter([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def multiproc_torch(frames: List[np.ndarray], size: List[int]) -> np.ndarray:
    rgbs = []
    for frame in frames:
        if frame is None:
            raise ValueError('Get empty frame.')
        fp32 = frame.astype(np.float32)
        resized = cv2.resize(fp32, size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgbs.append(rgb)

    rgb_batch = np.stack(rgbs, axis=0)
    norm_batch = rgb_batch/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_batch = (norm_batch - mean) / std
    chw_batch = np.transpose(norm_batch, (0, 3, 1, 2))

    return chw_batch


def torch_processor(frame: np.ndarray, size: List[int]) -> np.ndarray:
    fp32 = frame.astype(np.float32)
    resized = cv2.resize(fp32, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = torch_offsetter(normalizer(rgb))
    chw = np.transpose(norm, (2, 0, 1))
    return chw


def torch_proc(frame: np.ndarray, size: list) -> np.ndarray:
    # Resize
    _img = frame.astype(np.float32)
    resized = cv2.resize(_img, size[::-1])
    # Torch preprocess
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = rgb/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(3):
        norm[:, :, i] = (norm[:, :, i]-mean[i])/std[i]
    # Layout: hwc -> # chw
    chw = np.transpose(norm, (2, 0, 1))
    bchw = np.expand_dims(chw, axis=0)
    return bchw
