import ctypes
import threading
from typing import List, Tuple
from flask import Flask, render_template, request, redirect, send_file
import requests
import json
import numpy
import cv2
import time
from multiprocessing import Process
import multiprocessing
import io
import base64
app = Flask(__name__)


INFERENCE_SERVER_ADDR = 'http://127.0.0.1:9001'
WIDTH = 640
HEIGHT = 480

last_img_buf = multiprocessing.Array(ctypes.c_char, WIDTH * HEIGHT * 3)
buffer_filled = multiprocessing.Value(ctypes.c_bool, False)
mutex = None
vid = None


@app.route('/')
def lego():
    classes = json.loads(requests.get(INFERENCE_SERVER_ADDR + "/get-classes").text)
    return render_template('home.html', classes=classes)


@app.route('/process-img')
def process_img():
    img = copy_over_last_img()
    result, encoded_image = cv2.imencode(".png", img)
    if not result:
        return "Failed to encode image", 500
    else:
        return requests.post(INFERENCE_SERVER_ADDR + "/process-img", 
            files=[('image', ('image.png', encoded_image, 'image/png'))]).text


def copy_over_last_img() -> numpy.ndarray:
    # Get last image from shared buffer
    mutex.acquire()
    numpy_array_last_img = numpy.frombuffer(last_img_buf[:], dtype=numpy.uint8)
    mutex.release()

    # Give the array its old shape back
    last_img = numpy.reshape(numpy_array_last_img, newshape=(HEIGHT, WIDTH, 3))
    return last_img


def reinit_camera() -> None:
    global vid
    # define a video capture object
    if vid:
        vid.release()

    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #vid.set(cv2.CAP_PROP_SETTINGS, 0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def camera_loop(img_array: multiprocessing.Array, last_img_lock: threading.Lock) -> None:
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

    # Run camera loop
    camera_loop(img_array, last_img_lock)

    # After the loop release the cap object
    if vid:
        vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mutex = multiprocessing.Manager().Lock()
    p = Process(target=run_camera, args=(last_img_buf, mutex,))

    # Start camera process
    p.start()

    # Run web server
    app.run('127.0.0.1', 9000, debug=False)

    # Wait for camera process to end
    p.join()