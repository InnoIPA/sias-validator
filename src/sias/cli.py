import argparse
import os

from sias import build
from sias import model


def build_mode(args):
    onnx_path = args.onnx
    engine_path = args.engine
    if engine_path is None:
        path, ext = os.path.splitext(onnx_path)
        engine_path = f'{path}.engine'

    build.build_siamese_engine(
        onnx_file_path=onnx_path,
        engine_file_path=engine_path)


def run_mode(args):
    model_path = args.model
    input_files = args.inputs.split(',')
    if len(input_files) != 2:
        print('Error: Please provide two input file paths separated by a comma.')
    else:
        print(f'Running with model file: {model_path}')
        print(f'Input files: {input_files[0]}, {input_files[1]}')

    if model_path.endswith('.onnx'):
        siam = model.OnnxSiamese(model_path=model_path)
    else:
        siam = model.TrtSiamese(model_path=model_path)

    inputs = model.load_frames(siamese=siam, golden_paths=[
                               input_files[0]], sample_paths=[input_files[1]])
    outputs = siam.inference(inputs)
    print(f"Result: {outputs[0][0]}")


def build_args(subparsers):
    build_parser = subparsers.add_parser('build', help='Build mode')
    build_parser.add_argument('--onnx', required=True,
                              help='Path to ONNX file')
    build_parser.add_argument(
        '--engine', required=False, help='Path to Engine file')
    # build_parser.add_argument('--batchs', required=False, type=int, help='The maximum batch size')
    build_parser.set_defaults(func=build_mode)


def run_args(subparsers):
    run_parser = subparsers.add_parser('run', help='Run mode')
    run_parser.add_argument('--model', required=True,
                            help='Path to model file')
    run_parser.add_argument('--inputs', required=True,
                            help='Two input file paths separated by a comma')
    run_parser.set_defaults(func=run_mode)


def main():
    parser = argparse.ArgumentParser(
        description='Sample argparse with subcommands')
    subparsers = parser.add_subparsers(
        dest='mode', required=True, help='Select mode: build or run')

    build_parser = subparsers.add_parser('build', help='Build mode')
    build_parser.add_argument('--onnx', required=True,
                              help='Path to ONNX file')
    build_parser.add_argument(
        '--engine', required=False, help='Path to Engine file')
    build_parser.add_argument(
        '--batchs', required=False, type=int, help='The maximum batch size')
    build_parser.set_defaults(func=build_mode)

    run_parser = subparsers.add_parser('run', help='Run mode')
    run_parser.add_argument('--model', required=True,
                            help='Path to model file')
    run_parser.add_argument('--inputs', required=True,
                            help='Two input file paths separated by a comma')
    run_parser.set_defaults(func=run_mode)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
