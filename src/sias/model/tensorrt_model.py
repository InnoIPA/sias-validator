import sys
import os
import time
from typing import Dict, List
import numpy as np
import tensorrt as trt

from .model import BasicSiameseIO, load_frames
from . import common

os.environ["CUDA_MODULE_LOADING"] = "LAZY"


class TrtSiamese(BasicSiameseIO):

    def __init__(self, model_path: str, device: int = 0) -> None:

        self.trt_logger = trt.Logger()

        self.device = device
        self.engine = common.get_engine(model_path, self.trt_logger)
        self.input_shapes, self.output_shapes = \
            common.get_engine_info(self.engine)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream \
            = common.allocate_buffers(self.engine)

        super().__init__(self.input_shapes, self.output_shapes)

    def inference(self, inputs: Dict[str, List[np.ndarray]]):
        blob = self.preprocess(inputs=inputs)
        (input_1, input_2) = tuple(blob.values())
        self.inputs[0].host = input_1
        self.inputs[1].host = input_2

        outs = common.do_inference_v2(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        res = self.postprocess(outputs=outs)
        return res

    def __del__(self):
        try:
            common.free_buffers(self.inputs, self.outputs, self.stream)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":

    model_path = "/workspace/model/oi_model.engine"
    golden_path = "/workspace/4M4SQAA6Z1A10/data/val/9R4275R05010/golden/SolderLight/e14562e4-a584-4e01-9b46-87d3471cbfe2_c3819bb0-8783-4c41-a166-7e2dfb7c4a2b_40396183-d868-46d7-8d26-bfb9b935ed8a.jpg"
    sample_path = "/workspace/4M4SQAA6Z1A10/data/val/9R4275R05010/input/PASS/SolderLight/e14562e4-a584-4e01-9b46-87d3471cbfe2_c3819bb0-8783-4c41-a166-7e2dfb7c4a2b_40396183-d868-46d7-8d26-bfb9b935ed8a.jpg"

    print(f"""
    ---
    Model: {model_path}
    Folder: {os.path.dirname(golden_path)}
    ---
    UID: {os.path.splitext(os.path.basename(golden_path))[0]}
    Golden: {os.path.basename(golden_path)}
    Sample: {os.path.basename(sample_path)}
    ---
    """)

    siam = TrtSiamese(model_path=model_path)
    inputs = load_frames(
        siamese=siam,
        golden_paths=[golden_path],
        sample_paths=[sample_path])

    output = siam.inference(inputs)
    print(output[0][0])
