import os
import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

import argparse
import pickle
from collections import deque
import face_recognition

track_history = defaultdict(lambda: [])
model = YOLO("models/yolov9c.pt")
# make the model only detect person
model.classes = [0]
model.conf = 0.75


# # connecting setup of ip camera
# username = 'admin'
# password = 'Hikvision07!'
# ip = '192.169.0.100'
# rtsp_port = '554'

# # connect to ip camera
# cap = cv2.VideoCapture(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/h264/ch1/main/av_stream")




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--output", required=True,
	help="path to serialized db of facial dataset")
args = vars(ap.parse_args())
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
# time.sleep(2.0)



# Constants
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)
MAX_TRACK_HISTORY = 30
NUM_IMAGES_TO_SAVE = 10

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, 
                                      minNeighbors=MIN_NEIGHBORS, minSize=MIN_SIZE)
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    return boxes

def recognize_faces(rgb, boxes):
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    return names

def save_images(total):
    for i in range(NUM_IMAGES_TO_SAVE):
        p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        total += 1
    return total

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False, classes=[0])
        boxes = results[0].boxes.xyxy
        if results[0].boxes.id is not None:
            clss = results[0].boxes.cls.tolist()
            track_ids = results[0].boxes.id.int().tolist()
            annotator = Annotator(frame, line_width=2)
            for box, cls, track_id in zip(boxes, clss, track_ids):
                if cls == 0:
                    track = track_history.get(track_id, deque(maxlen=MAX_TRACK_HISTORY))
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    track_history[track_id] = track
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_boxes = detect_faces(frame)
                    names = recognize_faces(rgb, face_boxes)
                    #put the tracked point
                    total = 0
                    # Draw tracking lines and labels
                    for track_id, points in track_history.items():
                        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.circle(frame, points[-1][0], 7, (255, 0, 0), -1)
                        cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)
                    for ((top, right, bottom, left), name) in zip(face_boxes, names):
                        annotator.box_label(box, color=colors(int(cls), True), label=f"person (ID: {track_id} {name})")
                        if name == "Unknown":
                            total = save_images(total)
                                

        # result.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break



cap.release()
cv2.destroyAllWindows()
