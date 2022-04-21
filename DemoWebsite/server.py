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


WIDTH = 640
HEIGHT = 480

last_img_buf = multiprocessing.Array(ctypes.c_char, WIDTH * HEIGHT * 3)
buffer_filled = multiprocessing.Value(ctypes.c_bool, False)
mutex = None
vid = None

device = None
model = None

@app.route('/')
def lego():
    return render_template('home.html', classes=model.names)

@app.route('/process-img')
def process_img():
    last_img = copy_over_last_img()
    result_obj = run_inference(last_img)
    
    return json.dumps(result_obj)


def copy_over_last_img() -> numpy.ndarray:
    # Get last image from shared buffer
    mutex.acquire()
    numpy_array_last_img = numpy.frombuffer(last_img_buf[:], dtype=numpy.uint8)
    mutex.release()

    # Give the array its old shape back
    last_img = numpy.reshape(numpy_array_last_img, newshape=(HEIGHT, WIDTH, 3))
    return last_img


def run_inference(img0: numpy.ndarray) -> dict:
    global model, device

    # Resize
    img = letterbox(img0, (640, 640), stride=model.stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = numpy.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255
    if len(img.shape) == 3:
        img = img[None]
    
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.8, 0.45, None, False, max_det=1000)
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
    result, buffer = cv2.imencode(".png", result_img)

    result_dict = dict()
    result_dict['boxes'] = objects
    if result:
        result_dict['img'] = "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")

    return result_dict


def reinit_camera() -> None:
    global vid
    # define a video capture object
    if vid:
        vid.release()

    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #vid.set(cv2.CAP_PROP_SETTINGS, 0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def main_loop(img_array: multiprocessing.Array, last_img_lock: threading.Lock) -> None:
    global vid

    while True:
        ret = None
        frame = None

        while not ret:
            # Get a frame from the camera
            ret, frame = vid.read()

            # if error, try reinitializing
            if not ret:
                print("Encountered camera error. Trying to reinitialize...")
                reinit_camera()
                time.sleep(1)

        # Convert captured frame to a byte array
        frame_bytes = frame.tobytes('C')

        # Copy to shared buffer
        last_img_lock.acquire()
        img_array[:] = frame_bytes
        last_img_lock.release()

        if not buffer_filled.value:
            print("Camera ready, buffer filled.")
            buffer_filled.value = True

        # Resize frame for display
        #frame75 = cv2.resize(frame, (852, 480), interpolation=cv2.INTER_AREA)

        # Display the resulting frame
        #cv2.imshow('frame', frame75)

        #key = cv2.waitKey(1)
        #if key & 0xFF == ord('q'):
        #    break
        
        time.sleep(0.2)


def run_camera(img_array: multiprocessing.Array, last_img_lock: threading.Lock) -> None:
    global vid

    # Initialize camera
    reinit_camera()

    # Run mainloop
    main_loop(img_array, last_img_lock)

    # After the loop release the cap object
    if vid:
        vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


def setup_inference(weights: str) -> None:
    global model, device
    device = select_device()
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (640, 640)

    model.warmup(imgsz=(1, 3, *imgsz))


if __name__ == "__main__":
    setup_inference("weights3.pt")

    mutex = multiprocessing.Manager().Lock()
    p = Process(target=run_camera, args=(last_img_buf, mutex,))

    # Start camera process
    p.start()

    # Run web server
    app.run('127.0.0.1', 9001, debug=False)

    # Wait for camera process to end
    p.join()