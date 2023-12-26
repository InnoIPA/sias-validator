from typing import Dict, List
import numpy as np
import onnxruntime as ort
from .model import BasicSiameseIO, load_frames


class OnnxSiamese(BasicSiameseIO):

    def __init__(self, model_path: str) -> None:
        self.model = ort.InferenceSession(model_path)
        inputs = self.model.get_inputs()
        self.input_shapes = {
            input.name: input.shape for input in self.model.get_inputs()}
        self.output_shape = {
            output.name: output.shape for output in self.model.get_outputs()}
        super().__init__(self.input_shapes, self.output_shape)

    def inference(self, inputs: Dict[str, List[np.ndarray]]):
        blob = self.preprocess(inputs=inputs)
        outs = self.model.run(list(self.output_shape.keys()), blob)
        res = self.postprocess(outputs=outs)
        return res


if __name__ == "__main__":
    import os

    onnx_path = "/workspace/model/oi_model.onnx"
    golden_path = "/workspace/4M4SQAA6Z1A10/data/val/9R4275R05010/golden/SolderLight/e14562e4-a584-4e01-9b46-87d3471cbfe2_c3819bb0-8783-4c41-a166-7e2dfb7c4a2b_40396183-d868-46d7-8d26-bfb9b935ed8a.jpg"
    sample_path = "/workspace/4M4SQAA6Z1A10/data/val/9R4275R05010/input/PASS/SolderLight/e14562e4-a584-4e01-9b46-87d3471cbfe2_c3819bb0-8783-4c41-a166-7e2dfb7c4a2b_40396183-d868-46d7-8d26-bfb9b935ed8a.jpg"

    print(f"""
    ---
    Model: {onnx_path}
    Folder: {os.path.dirname(golden_path)}
    ---
    UID: {os.path.splitext(os.path.basename(golden_path))[0]}
    Golden: {os.path.basename(golden_path)}
    Sample: {os.path.basename(sample_path)}
    ---
    """)

    siam = OnnxSiamese(
        model_path=onnx_path
    )
    inputs = load_frames(
        siamese=siam,
        golden_paths=[golden_path],
        sample_paths=[sample_path])

    output = siam.inference(inputs)
    print(output[0][0])
