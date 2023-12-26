#!/usr/bin/python3
import csv
import json
import os
import sys
import datetime
import logging
import logging
import os
import colorlog
from collections import defaultdict

from tqdm import tqdm

# Custom
import sias


def get_logger(root: str = "/workspace/logs"):

    logger_path = os.path.join(root, f"{datetime.datetime.now()}.log")
    if not os.path.exists(root):
        os.mkdir(root)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    console_formatter = colorlog.ColoredFormatter(
        "%(asctime)s %(log_color)s [%(levelname)-.4s] %(reset)s %(message)s ", "%y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-.4s] %(message)s", "%y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def read_csv(file_path: str) -> list:
    data = []
    with open(file_path, newline='') as csvfile:
        for row in csv.reader(csvfile):
            data.append(row)
        return data


def read_json(file_path: str) -> dict:
    with open(file_path, 'r') as jsonfile:
        data = json.load(jsonfile)
        return data


def build_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        help="the csv file path.", required=True)
    parser.add_argument("--json", type=str,
                        help="the json file path.", required=True)
    parser.add_argument("--data", type=str,
                        help="the validation data root.", required=True)
    parser.add_argument("--model", type=str,
                        help="the model path.", required=False)
    return parser.parse_args()


def main(args):

    def get_path(path): return os.path.join(args.data, path)
    def is_exist(path): return os.path.exists(path)

    assert is_exist(args.data), f"{args.data} not found."
    assert is_exist(args.csv), f"{args.csv} not found."

    # Logger
    logger = get_logger()

    # Load Model
    if args.model.endswith(".onnx"):
        siam = sias.model.OnnxSiamese(model_path=args.model)
        logger.info('Loaded ONNX Model !')
    else:
        siam = sias.model.TrtSiamese(model_path=args.model)
        logger.info('Loaded TensorRT Engine !')

    # Read metric
    csv_data = read_csv(args.csv)
    json_data = read_json(args.json)

    valid_data, unvalid_data, missing_images, missing_goldens = [], [], [], []
    metric = defaultdict(lambda: defaultdict(int))
    total_images_num = len(csv_data[1:])
    for row in tqdm(csv_data[1:], desc='Processing', unit="file"):

        # Parse data ( for inference.csv )
        [sample_path, golden_path, label,
            object_name, siamese_score] = row
        golden_uid = '_'.join(object_name.split('_')[:2])
        thres = json_data[golden_uid]["threshold"]
        thres, siamese_score = map(float, (thres, siamese_score))

        # Get correct path
        def get_path(path):
            return f"{os.path.join(args.data, path, object_name)}.jpg"
        golden_path, sample_path = map(get_path, (golden_path, sample_path))

        # If golden not found
        if golden_uid not in json_data:
            missing_goldens.append(golden_uid)
            continue
        # If file not found
        check_path = { path:os.path.isfile(path) for path in [golden_path, sample_path] }
        for path, exist in check_path.items():
            if exist: continue
            missing_images.append(path)
        if False in check_path.values():
            continue

        # Preprare data
        inputs = sias.model.load_frames(
            siamese=siam,
            golden_paths=[golden_path],
            sample_paths=[sample_path])

        # Do inference
        output = siam.inference(inputs=inputs)[0][0]
        
        flag = label if output == thres else 'PASS' if output < thres else 'NG'
        
        # PASS: True, False ; NG: True, False ( NG False is Negative )
        metric[label][flag=='PASS'] += 1
        
        # Tidy up data
        (valid_data if flag == label else unvalid_data).append({
            'name': object_name,
            'label': label,
            'gt': siamese_score,
            'thres': thres,
            'predict': output,
            'flag': flag,
            'diff': siamese_score-output
        })

    # Record infor
    (missing_goldens_num, missing_images_num) = map(len, (missing_goldens, missing_images))
    (valid_num, unvalid_num) = map(len, (valid_data, unvalid_data))
    total_num = valid_num + unvalid_num
    logger.info('--- START ---')
    logger.info(f'Total Images: {total_images_num}') 
    logger.info(f'Missing Images: {missing_images_num}')
    logger.info(f'Missing Golden: {missing_goldens_num}')
    
    logger.info('--- INFERENCE ---')
    logger.info(f'Total Request: {total_num}')
    logger.info(f'Valid (matched GT): {valid_num}')
    logger.info(f'Unvalid (unmatched GT): {unvalid_num}')

    logger.info('--- METRIC ---')
    # logger.debug('Acc: (TP+TN)/(TP+FP+TN+FN)')
    # logger.debug('Recall: (TP)/(TP+FN)')
    # logger.debug('FPR: (FP)/(TP+FP)')
    # logger.debug('Defect Rate: (FN)/(TN+FN)')
    (tp, fp), (tn, fn) = metric['PASS'].values(), metric['NG'].values()
    logger.info(f'Accuracy: {((tn+tp)/(tn+fn+tp+fp))*100:03.1f}%')
    logger.info(f'Recall: {((tn)/(tn+fp))*100:03.1f}%')
    logger.info(f'False Positive Rate (FPR): {(fp/(fp+tp))*100:03.1f}%')
    logger.info(f'Defect Rate: {(tn/(tn+fn))*100:03.1f}%')

    logger.debug('--- DETAIL ---')
    for idx, data in enumerate(unvalid_data):
        logger.debug(f'[{idx+1}]')
        for key, val in data.items():
            logger.debug(f"{key}: {val}")

if __name__ == "__main__":
    """
    inference.csv: ['input_path', 'golden_path', 'label', 'object_name', 'siamese_score'] 

    Usage: python3 validator.py --csv ./model/inference.csv --root 4M4SQAA6Z1A10/data/val --model ./model/oi_model.engine --json ./model/golden_info.json
    """
    main(build_args())
