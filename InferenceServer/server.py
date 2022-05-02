import ctypes
import threading
from typing import List, Tuple
from flask import Flask, render_template, request, redirect, send_file
app = Flask(__name__)
import numpy
import cv2
import time
from multiprocessing import Process
import multiprocessing
import io
import base64

import json

import torch

from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

WIDTH = 1280
HEIGHT = 720

APPROVED_CLASSES = ["10247", "2357", "2420", "2458", "2495", "3023", "3031", "3040",
                    "3710", "32064", "3794", "3795", "3894", "4079", "41770", "44728",
                    "4865", "6143", "61780", "6541", "98302"]

model = None
device = None


@app.route('/process-img', methods=['POST'])
def process_img():
    start = time.time()
    img_file = request.files['image']
    img = cv2.imread(img_file)
    result_obj = run_inference(img)
    print("inference took", time.time() - start)
    
    return json.dumps(result_obj)


@app.route('/get-classes', methods=['GET'])
def get_classes():
    result = list(model.names)
    return json.dumps(result)


def run_inference(img0: numpy.ndarray) -> dict:
    global model, device

    # img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # Resize
    img = letterbox(img0, (1024, int(1024 * (HEIGHT / WIDTH))), stride=model.stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = numpy.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255
    if len(img.shape) == 3:
        img = img[None]
    
    pred = model(img, augment=False, visualize=False)
    #approved_class_indices = [model.names.index(x) for x in APPROVED_CLASSES]
    pred = non_max_suppression(pred, 0.35, 0.45, None, False, multi_label=True, max_det=1000)

    det = pred[0]
    gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
    annotator = Annotator(img0.copy(), line_width=3, example=str(model.names))

    objects = []

    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            c = int(cls)

            obj = dict()
            obj['class_name'] = model.names[c]
            obj['conf'] = float(conf)
            obj['x'] = xywh[0]
            obj['y'] = xywh[1]
            obj['w'] = xywh[2]
            obj['h'] = xywh[3]

            objects.append(obj)

            #label = f'{model.names[c]} {conf:.2f}'
            label = f'{model.names[c]}'
            annotator.box_label(xyxy, label, color=colors(c, True))

    result_img = annotator.result()    
    result_img = cv2.resize(result_img, (854, int(854 * (HEIGHT / WIDTH))), interpolation=cv2.INTER_AREA)
    result, buffer = cv2.imencode(".jpg", result_img)

    result_dict = dict()
    result_dict['boxes'] = objects
    if result:
        result_dict['img'] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    return result_dict


def setup_inference(weights: str) -> None:
    global model, device
    device = select_device()
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (1024, 1024)

    model.warmup(imgsz=(1, 3, *imgsz))


if __name__ == "__main__":
    setup_inference("weights5.pt")
    app.run("127.0.0.1", 9001, debug=False)

