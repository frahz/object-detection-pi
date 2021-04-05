# necessary packages
from flask import Response, Flask, render_template
from imutils.video import VideoStream
from object_detection import ObjectDetection
import numpy as np
import threading
import argparse
import imutils
import time
import cv2

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def detection(resolution, use_TPU):
    global vs, outputFrame, lock

    obj_detect = ObjectDetection(resolution=resolution, use_TPU=use_TPU)

    interpreter = obj_detect.interpreter
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    while True:
        # start timer for calculating framerate
        t1 = cv2.getTickCount()

        # grab frame and resize to expected shape [1xHxWx3]
        frame1 = vs.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # Retrieve detection results
        # Bounding box coordinates of detected objects
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        obj_detect.draw_detection(frame, boxes, classes, scores)

        # draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        with lock:
            outputFrame = frame.copy()


vs = VideoStream(usePiCamera=1, resolution=(944, 528)).start()
time.sleep(2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='944x528')
    args = parser.parse_args()
    
    use_TPU = args.edgetpu
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    resolution = (imW, imH)

    t = threading.Thread(target=detection, args=(resolution, use_TPU))
    t.daemon = True
    t.start()

    app.run(host="192.168.1.91", port="8000")

vs.stop()
