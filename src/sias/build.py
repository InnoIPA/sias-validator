
import os
import sys
import tensorrt as trt

sys.path.append(os.path.dirname(__file__))
from .model import common

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
TRT_LOGGER = trt.Logger()


def build_siamese_engine(onnx_file_path, engine_file_path, max_batch_size=1):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        common.EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        config.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = max_batch_size
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print("ONNX file {} not found.".format(onnx_file_path))
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # FIXME: Fix the batch size ( only input layer could be change )
        # input_1, input_2 = (network.get_input(0).shape, network.get_input(1).shape)
        # if builder.max_batch_size is None:
        #     builder.max_batch_size = input_1[0]
        # network.get_input(0).shape = [builder.max_batch_size, input_1[1], input_1[2], input_1[3]]
        # network.get_input(1).shape = [builder.max_batch_size, input_2[1], input_2[2], input_2[3]]
        
        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(
            onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        print(f"Generated Engine File: {engine_file_path}")
        print("\nSuccessed !")
        return engine


def build_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str,
                        help="the model path.", required=True)
    parser.add_argument("--engine", type=str,
                        help="the model path.", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    onnx_path = args.onnx
    engine_path = args.engine
    if engine_path is None:
        path, ext = os.path.splitext(onnx_path)
        engine_path = f'{path}.engine'
    build_siamese_engine(
        onnx_file_path=onnx_path,
        engine_file_path=engine_path)
    print(onnx_path, engine_path)
